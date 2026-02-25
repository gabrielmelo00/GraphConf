# Sync the python environment
sync:
  uv sync --all-packages

# Symlink packages in `Model` for editor autocompletion
symlink:
  ln -s ../../../../Models/SpecBridge/specbridge/ .venv/lib/python3.11/site-packages/
  ln -s ../../../../Models/SpecBridge/DreaMS/dreams/ .venv/lib/python3.11/site-packages/
  ln -s ../../../../Models/Any2Graph/Any2Graph .venv/lib/python3.11/site-packages/
  ln -s ./fngw .venv/lib/python3.11/site-packages/
