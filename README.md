# Vrishabhavathi Valley Watershed Project

AI-enabled circular water resource management for the Vrishabhavathi Valley watershed, Bengaluru.

## Setup (Windows, PowerShell)

### 1. Install these first
- Git: https://git-scm.com/download/win
- Miniforge (Windows x86_64): https://github.com/conda-forge/miniforge (download the `.exe` installer)
- Docker Desktop: https://www.docker.com/products/docker-desktop (keep the WSL 2 backend option ticked in the installer; you will not need to use WSL yourself)
- QGIS LTR: https://qgis.org (use the OSGeo4W installer, Advanced Install, and tick **QGIS LTR**, **GRASS GIS** and **GDAL**)
- Optional: VS Code

Restart your PC after installing Docker Desktop.

### 2. One-time conda setup
Open **Miniforge Prompt** from the Start menu and run:

    conda init powershell

Close it, then open a new **PowerShell** window.

### 3. Clone the repo
Use a folder without spaces and outside OneDrive, for example `C:\projects`:

    mkdir C:\projects
    cd C:\projects
    git clone https://github.com/S-Farhan-Hashmi/MAIN-EL.git
    cd MAIN-EL

### 4. Create the Python environment

    mamba env create -f environment.yml
    mamba activate vv-watershed

Check that it works:

    python -c "import geopandas, rasterio, xgboost, pulp; print('ok')"

Do not press Ctrl+C while it installs. It can take 5 to 10 minutes.

### 5. Database (PostGIS via Docker)
Make sure Docker Desktop is running, then:

    copy .env.example .env
    notepad .env

Replace `your_password_here` with a password you choose and save. Then:

    docker compose up -d
    docker exec -it vv-postgis psql -U postgres -d watershed -c "CREATE EXTENSION IF NOT EXISTS postgis; SELECT postgis_version();"

If you see a PostGIS version number, it works.

### 6. QGIS
Open QGIS, go to Processing > Toolbox and search for `r.watershed`. If it appears under GRASS, you are set.

## Folder layout
- `data/` local only, never committed
- `src/` code (ingest, features, models, dashboard)
- `notebooks/` exploration
- `gis/` QGIS projects and notes
- `docs/assumptions.md` running list of assumptions and limitations

## Team rules
- Never commit data files or `.env`
- Work on your own branch: `git checkout -b yourname/task`, then open a pull request into `main`
- If you add a package, add it to `environment.yml` and tell the team
- Share large datasets through the shared Drive folder and note sources in `docs/assumptions.md`
- Label any simulated data as simulated everywhere it appears

## Common problems
- **Docker says WSL 2 is incomplete or shows a virtualization error:** enable virtualization in BIOS, run `wsl --update` in PowerShell, then restart.
- **mamba cannot lock the package directory:** another conda process is running. Close other terminals or restart the PC, then retry.
- **`conda` or `mamba` not recognized in PowerShell:** redo step 2, or use Miniforge Prompt instead.
- **Port 5432 already in use:** a local PostgreSQL is running. Stop it, or change the port in `.env` and `docker-compose.yml`.
- **Docker permission or "not running" errors:** open Docker Desktop and wait until it says it is running.
