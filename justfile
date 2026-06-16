# Sync the python environment
sync:
  uv sync --all-packages

# Symlink packages in `Model` for editor autocompletion
symlink:
  ln -s ../../../../Models/Any2Graph/Any2Graph .venv/lib/python3.11/site-packages/
  ln -s ./fngw .venv/lib/python3.11/site-packages/
