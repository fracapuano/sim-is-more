# !bin/bash

echo "Building the necessary tables"

cmds=("python commons/tables/build_hw_nats_dataset.py" "python commons/tables/build_hw_nats_dataset.py")
for cmd in "${cmds[@]}"; do
  $cmd &
done

wait