# List available recipes by default
default:
    @just --list

# Prepare everything. Requires an elevation .tif and an OSM extract (.osm.pbf)
prep tif pbf:
    #!/usr/bin/env bash
    set -euxo pipefail
    realtif=`readlink -f {{tif}}`
    sed -i -e "s|YOURPATH|$realtif|g" ./server/main.py
    just assets
    just install
    just tile {{pbf}}

# Fetch assets
assets:
    mkdir -p assets
    wget https://www.techmonkeybusiness.com/galleries/Texture_Galleries/Billboard_Trees/images/001-Bigtree1.png -O assets/tree0.png 
    wget https://www.techmonkeybusiness.com/galleries/Texture_Galleries/Billboard_Trees/images/000-bigtree2.png -O assets/tree0.png 

# install server tools, just clean to uninstall
[working-directory("server")]
install:
    cargo install martin

    uv venv
    source .venv/bin/activate
    uv pip install uvicorn titiler.application
    
# remove system-wide tools
clean:
    cargo uninstall martin

# Regenerate vector tiles using tilemaker (requires docker)
[working-directory("server")]
tile input='germany-latest.osm.pbf' output='germany_buildings.pmtiles':
    docker run -it --rm --pull always -v $(pwd):/data -w /data \
        ghcr.io/systemed/tilemaker:master \
        /data/{{input}} \
        --output /data/{{output}} \
        --config /data/config.json \
        --process /data/process.lua

# Start tileserver (requires tmux)
[working-directory("server")]
serve:
    # 1. Create new window named 'tileservers'
    tmux new-window -n tileservers
    
    # 2. Split horizontal to create the Right pane (Interactive)
    tmux split-window -h -t tileservers
    
    # 3. Select Left pane (0) and split it vertical
    tmux select-pane -t tileservers.0
    tmux split-window -v -t tileservers.0
    
    # 4. Send commands
    # Top Left (Pane 0) -> Martin
    tmux send-keys -t tileservers.0 'martin -c martin_config.yaml' C-m
    
    # Bottom Left -> Uvicorn
    # We select Pane 0, move Down to find the correct split, then send keys
    tmux select-pane -t tileservers.0
    tmux select-pane -D 
    tmux send-keys 'source .venv/bin/activate && uvicorn main:app --reload' C-m
    
    # 5. Focus on the Right pane (Interactive) for your use
    tmux select-pane -t tileservers.0
    tmux select-pane -R

# Stop tileserver
stop:
    tmux kill-window -t tileservers
