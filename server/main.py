from fastapi import FastAPI
from titiler.core.factory import TilerFactory

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development - be more specific in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fixed_path(url: str = None):
    return "/mnt/ubuntu/home/casimir/UNI/WS_25/testing/osmviz/euro-dem-tif/data/eurodem.tif"

# Create a TilerFactory for Cloud-Optimized GeoTIFFs
cog = TilerFactory(path_dependency=fixed_path)

# Register all the COG endpoints automatically
app.include_router(cog.router, tags=["Cloud Optimized GeoTIFF"])


# Optional: Add a welcome message for the root endpoint
@app.get("/")
def read_index():
    return {"message": "Welcome to TiTiler"}
