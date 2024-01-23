use godot::{
    bind::{godot_api, GodotClass},
    engine::{ISprite2D, Sprite2D},
    log::godot_print,
    obj::{Base, WithBaseField},
};
use godot::builtin::GString;
use godot::engine::{INode, Node};
use godot::obj::{Bounds, GodotClass};

#[derive(Debug, GodotClass)]
#[class(base = Sprite2D)]
pub struct RustPlayer {
    #[base]
    base: Base<Sprite2D>,
    speed: f64,
    angular_speed: f64,
}

#[godot_api]
impl ISprite2D for RustPlayer {
    fn init(base: Base<Sprite2D>) -> Self {
        godot_print!("Hello, world!"); // Prints to the Godot console

        Self { speed: 400.0, angular_speed: std::f64::consts::PI, base }
    }
    fn physics_process(&mut self, delta: f64) {
        // In GDScript, this would be:
        // rotation += angular_speed * delta

        let radians = (self.angular_speed * delta) as f32;
        self.base_mut().rotate(radians);
        // The 'rotate' method requires a f32,
        // therefore we convert 'self.angular_speed * delta' which is a f64 to a f32
    }
}

#[derive(Debug, GodotClass)]
#[class(base = Node)]
pub struct CellAnchor {
    #[base]
    base: Base<Node>,
    #[export]
    bed_rock_layer: GString,
    #[export]
    block_layer: GString,
    name: String,
    hit_points: i32,
}

#[godot_api]
impl INode for CellAnchor {
    fn init(base: Base<Self::Base>) -> Self {
        Self { base, bed_rock_layer: Default::default(), block_layer: Default::default(), name: String::new(), hit_points: 0 }
    }
}


#[derive(Debug, GodotClass)]
#[class(base = Node)]
pub struct CellEditor {
    #[base]
    base: Base<Node>,
    name: String,
    hit_points: i32,
}

#[godot_api]
impl INode for CellEditor {
    fn init(base: Base<Self::Base>) -> Self {
        Self { base, name: String::new(), hit_points: 0 }
    }
}