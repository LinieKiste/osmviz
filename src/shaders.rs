// src/shaders.rs

pub mod terrain_vs { vulkano_shaders::shader! { ty: "vertex", path: "shaders/terrain.vert", } }
pub mod tcs { vulkano_shaders::shader! { ty: "tess_ctrl", path: "shaders/terrain.tesc", } }
pub mod tes { vulkano_shaders::shader! { ty: "tess_eval", path: "shaders/terrain.tese", } }
pub mod terrain_fs { vulkano_shaders::shader! { ty: "fragment", path: "shaders/terrain.frag", } }

pub mod building_vs { vulkano_shaders::shader! { ty: "vertex", path: "shaders/buildings.vert", } }
pub mod building_fs { vulkano_shaders::shader! { ty: "fragment", path: "shaders/buildings.frag", } }

pub mod tree_vs { vulkano_shaders::shader! { ty: "vertex", path: "shaders/tree.vert" } }
pub mod tree_fs { vulkano_shaders::shader! { ty: "fragment", path: "shaders/tree.frag" } }

