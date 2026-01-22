use glam::{Mat4, Vec3, Vec4};
use std::f32::consts::PI;
use vulkano::buffer::{BufferContents, Subbuffer};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- 1. The Data Sent to the Shader ---
// This must match the layout(std140) in GLSL exactly.
// Alignment rules: vec3 acts like vec4 (16 bytes), mat4 is 64 bytes.
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub position: [f32; 3], 
    pub _padding: f32, // Essential for std140 alignment (vec3 is 16 bytes aligned)
}

// --- 2. The Logic Controller ---
pub struct Camera {
    // Spatial properties
    pub position: Vec3,
    pub yaw: f32,   // Horizontal angle (radians)
    pub pitch: f32, // Vertical angle (radians)

    // Lens properties
    pub aspect_ratio: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,

    // Movement state (for smooth input handling)
    speed: f32,
    rot_speed: f32,
    sensitivity: f32,
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,

    rotate_left: bool,
    rotate_right: bool,
    rotate_up: bool,
    rotate_down: bool,
}

impl Camera {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            // Start high up, looking down
            position: Vec3::new(0., 2., 2.), 
            yaw: -PI / 2.0, // Look along -Z
            pitch: -PI / 4.0, // Look down 45 degrees
            
            aspect_ratio: width / height,
            fov: 45.0_f32.to_radians(),
            near: 0.1,
            far: 2000.0,

            speed: 1.0, // Meters per second
            rot_speed: 1.0,
            sensitivity: 0.002,
            
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,

            rotate_left: false,
            rotate_right: false,
            rotate_up: false,
            rotate_down: false,
        }
    }

    pub fn get_zoom_level(&self) -> u8 {
        22-(self.position.y.max(1.0).ln() as u8).min(20)
    }

    /// Update internal aspect ratio when window resizes
    pub fn resize(&mut self, width: f32, height: f32) {
        self.aspect_ratio = width / height;
    }

    /// Handle Keyboard Inputs
    pub fn handle_input(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    state,
                    physical_key: PhysicalKey::Code(keycode),
                    ..
                },
            ..
        } = event
        {
            let is_pressed = *state == ElementState::Pressed;
            match keycode {
                // WASD movement
                KeyCode::KeyW => self.move_forward = is_pressed,
                KeyCode::KeyS => self.move_backward = is_pressed,
                KeyCode::KeyA => self.move_left = is_pressed,
                KeyCode::KeyD => self.move_right = is_pressed,

                // Rotation (Arrow Keys)
                KeyCode::ArrowLeft => self.rotate_left = is_pressed,
                KeyCode::ArrowRight => self.rotate_right = is_pressed,
                KeyCode::ArrowUp => self.rotate_up = is_pressed,
                KeyCode::ArrowDown => self.rotate_down = is_pressed,

                // Up/down
                KeyCode::Space => self.move_up = is_pressed,
                KeyCode::ShiftLeft => self.move_down = is_pressed,
                _ => {}
            };
        }
    }

    /// Update Position based on time delta
    pub fn update(&mut self, delta_time: f32) {
        // Calculate forward/right vectors based on yaw
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let forward = Vec3::new(cos_y, 0.0, sin_y).normalize();
        let right = Vec3::new(sin_y, 0.0, -cos_y).normalize();
        let up = Vec3::Y;

        let velocity = self.speed * delta_time;

        if self.move_forward { self.position += forward * velocity; }
        if self.move_backward { self.position -= forward * velocity; }
        if self.move_right { self.position -= right * velocity; }
        if self.move_left { self.position += right * velocity; }

        // Rotation Logic (Radians per second)
        if self.rotate_left { self.yaw -= self.rot_speed * delta_time; }
        if self.rotate_right { self.yaw += self.rot_speed * delta_time; }
        
        // Clamp Pitch to prevent flipping upside down
        if self.rotate_up { self.pitch += self.rot_speed * delta_time; }
        if self.rotate_down { self.pitch -= self.rot_speed * delta_time; }
        self.pitch = self.pitch.clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);

        if self.move_up { self.position += up * velocity; }
        if self.move_down { self.position -= up * velocity; }
    }

    /// Construct the matrices and uniform data
    pub fn get_uniform_data(&self) -> CameraUniform {
        // 1. View Matrix (LookAt)
        // Standard FPS camera math
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();

        let direction = Vec3::new(
            cos_y * cos_p,
            sin_p,
            sin_y * cos_p
        ).normalize();

        let view = Mat4::look_at_rh(self.position, self.position + direction, Vec3::Y);

        // 2. Projection Matrix (Perspective)
        let mut proj = Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far);
        
        // VULKAN FIX: Vulkan's Y-clip-space is inverted compared to OpenGL
        // We flip the Y scaling in the projection matrix to fix this.
        proj.y_axis.y *= -1.0;

        CameraUniform {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: self.position.to_array(),
            _padding: 0.0,
        }
    }

    /// Returns the world coordinate where the camera is looking at the ground (Y=0).
    /// Returns None if looking at the sky or horizon.
    pub fn intersect_ground(&self) -> Option<Vec3> {
        // 1. Reconstruct Forward Vector from Yaw/Pitch
        // (Same math used in get_uniform_data)
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();

        let forward = Vec3::new(
            cos_y * cos_p,
            sin_p,
            sin_y * cos_p
        ).normalize();

        // 2. Check if looking up or parallel (Y must be negative to hit ground)
        if forward.y >= -0.0001 {
            return None;
        }

        // 3. Ray-Plane Intersection (t = -Origin.y / Dir.y)
        let t = -self.position.y / forward.y;

        // 4. Calculate Hit Point
        Some(self.position + forward * t)
    }
}
