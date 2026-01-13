#!/bin/bash

# Script to generate metrics for a single Java file
# Input: Path to Java file
# Output: CSV file with metrics for the Java class, bug count (set to 0 since git log isn’t available)

# Check if Java file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-java-file>"
  exit 1
fi

JAVA_FILE="$1"
FILE_NAME=$(basename "$JAVA_FILE")
CLASS_NAME="${FILE_NAME%.*}"
OUTPUT_DIR="metrics_output"
CSV_FILE="$OUTPUT_DIR/${CLASS_NAME}_metrics.csv"
VENV_DIR="venv"

# Ensure required tools are installed
if ! command -v python3 &> /dev/null; then
  echo "Python3 is required but not installed. Please install python3."
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install javalang if not present
if ! python3 -c "import javalang" &> /dev/null; then
  echo "Installing javalang in virtual environment..."
  pip install javalang
  if [ $? -ne 0 ]; then
    echo "Failed to install javalang."
    deactivate
    exit 1
  fi
fi

# Create Python script for parsing and computing metrics
PYTHON_SCRIPT=$(mktemp)
cat << 'EOF' > "$PYTHON_SCRIPT"
import javalang
import sys
import csv
from collections import defaultdict

def get_cyclomatic_complexity(method):
    complexity = 1
    for _, node in method.filter(javalang.tree.IfStatement):
        complexity += 1
    for _, node in method.filter(javalang.tree.ForStatement):
        complexity += 1
    for _, node in method.filter(javalang.tree.WhileStatement):
        complexity += 1
    for _, node in method.filter(javalang.tree.DoStatement):
        complexity += 1
    for _, node in method.filter(javalang.tree.SwitchStatement):
        complexity += len([s for s in node.cases if s.statements])
    for _, node in method.filter(javalang.tree.CatchClause):
        complexity += 1
    return complexity

def get_method_parameters(method):
    return [param.type.name for param in method.parameters] if method.parameters else []

def calculate_cam(methods):
    if not methods or len(methods) < 2:
        return 0
    param_sets = [frozenset(get_method_parameters(m)) for m in methods]
    shared_params = 0
    total_comparisons = 0
    for i, p1 in enumerate(param_sets):
        for p2 in param_sets[i+1:]:
            if p1 and p2:
                shared_params += len(p1 & p2)
                total_comparisons += len(p1 | p2)
    return shared_params / total_comparisons if total_comparisons > 0 else 0

def analyze_file(file_path, project_name, version):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        tree = javalang.parse.parse(code)
    except:
        return None

    for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
        fully_qualified_name = f"{tree.package.name}.{class_node.name}" if tree.package else class_node.name
        metrics = {
            'project_name': project_name,
            'version': version,
            'class_name': fully_qualified_name,
            'wmc': 0,
            'rfc': 0,
            'loc': len(code.splitlines()),
            'max_cc': 0,
            'avg_cc': 0,
            'cbo': 0,
            'ca': 0,
            'ce': 0,
            'ic': 0,
            'cbm': 0,
            'lcom': 0,
            'lcom3': 0,
            'dit': 0,
            'noc': 0,
            'mfa': 0,
            'npm': 0,
            'dam': 0,
            'moa': 0,
            'cam': 0,
            'amc': 0,
            'bug': 0   # No git history for single file
        }

        # Methods and complexity
        methods = class_node.methods
        metrics['wmc'] = len(methods)
        cc_values = []
        method_names = set()
        for method in methods:
            cc = get_cyclomatic_complexity(method)
            cc_values.append(cc)
            method_names.add(method.name)
            if isinstance(method, javalang.tree.MethodDeclaration):
                metrics['npm'] += 1 if method.modifiers and 'public' in method.modifiers else 0
        metrics['max_cc'] = max(cc_values) if cc_values else 0
        metrics['avg_cc'] = sum(cc_values) / len(cc_values) if cc_values else 0
        metrics['amc'] = metrics['loc'] / metrics['wmc'] if metrics['wmc'] > 0 else 0

        # Inheritance metrics
        metrics['dit'] = 1 if class_node.extends else 0
        metrics['ic'] = metrics['dit']

        # Coupling and cohesion
        fields = [f for f in class_node.fields if isinstance(f, javalang.tree.FieldDeclaration)]
        metrics['moa'] = sum(1 for f in fields if f.type and isinstance(f.type, javalang.tree.ReferenceType))
        total_fields = len(fields)
        private_fields = sum(1 for f in fields if f.modifiers and ('private' in f.modifiers or 'protected' in f.modifiers))
        metrics['dam'] = private_fields / total_fields if total_fields > 0 else 0

        # LCOM calculation
        field_usage = defaultdict(set)
        for method in methods:
            for _, node in method.filter(javalang.tree.MemberReference):
                if node.qualifier in [f.declarators[0].name for f in fields]:
                    field_usage[method.name].add(node.qualifier)
        lcom = 0
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                if not (field_usage[m1.name] & field_usage[m2.name]):
                    lcom += 1
        metrics['lcom'] = lcom
        metrics['lcom3'] = 2 * lcom / (len(methods) * (len(methods) - 1)) if len(methods) > 1 else 0

        # RFC and CBO
        called_methods = set()
        for method in methods:
            for _, node in method.filter(javalang.tree.MethodInvocation):
                called_methods.add(node.member)
        metrics['rfc'] = len(methods) + len(called_methods)
        metrics['cbo'] = len(called_methods)

        # CBM: Count intra-class method calls
        intra_class_calls = 0
        for method in methods:
            for _, node in method.filter(javalang.tree.MethodInvocation):
                if node.member in method_names:
                    intra_class_calls += 1
        metrics['cbm'] = intra_class_calls

        # CAM: Cohesion among methods
        metrics['cam'] = calculate_cam(methods)

        return metrics
    return None

if __name__ == '__main__':
    java_file = sys.argv[1]
    project_name = sys.argv[2]
    version = sys.argv[3]
    output_csv = sys.argv[4]

    metrics = analyze_file(java_file, project_name, version)
    if not metrics:
        print("No classes found in file.")
        sys.exit(1)

    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['project_name', 'version', 'class_name', 'wmc', 'rfc', 'loc', 'max_cc', 'avg_cc',
                      'cbo', 'ca', 'ce', 'ic', 'cbm', 'lcom', 'lcom3', 'dit', 'noc', 'mfa',
                      'npm', 'dam', 'moa', 'cam', 'amc', 'bug']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Processed {metrics['class_name']}")
EOF

# Create CSV header
echo "Generating metrics for $CLASS_NAME..."
echo "project_name,version,class_name,wmc,rfc,loc,max_cc,avg_cc,cbo,ca,ce,ic,cbm,lcom,lcom3,dit,noc,mfa,npm,dam,moa,cam,amc,bug" > "$CSV_FILE"

# Run Python script to analyze Java file
python3 "$PYTHON_SCRIPT" "$JAVA_FILE" "$CLASS_NAME" "1.0" "$CSV_FILE"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && [ -f "$CSV_FILE" ]; then
  LINE_COUNT=$(wc -l < "$CSV_FILE")
  if [ "$LINE_COUNT" -gt 1 ]; then
    echo "Metrics generated successfully. Output saved to $CSV_FILE"
  else
    echo "Metrics generation failed: No classes processed."
    rm "$PYTHON_SCRIPT"
    deactivate
    exit 1
  fi
else
  echo "Failed to generate metrics. Python script exited with code $EXIT_CODE."
  rm "$PYTHON_SCRIPT"
  deactivate
  exit 1
fi

# Clean up
rm "$PYTHON_SCRIPT"
deactivate
