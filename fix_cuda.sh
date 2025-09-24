#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# fix_tf_cuda.sh
# Automatically discover NVIDIA CUDA library paths installed by pip
# and export them into LD_LIBRARY_PATH for TensorFlow GPU support.
# ------------------------------------------------------------------------------

set -e

# Find the base site-packages directory
SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")

# Collect all nvidia/*/lib paths
LIB_PATHS=$(python - <<'PY'
import site, glob, os
paths = set()
for base in site.getsitepackages():
    for p in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
        paths.add(p)
print(":".join(sorted(paths)))
PY
)

# If nothing found, warn
if [ -z "$LIB_PATHS" ]; then
  echo "No nvidia/*/lib directories found under $SITEPKG"
  exit 1
fi

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LIB_PATHS:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH updated:"
echo "$LD_LIBRARY_PATH"

# Optional: run a quick TensorFlow GPU check
python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
PY
