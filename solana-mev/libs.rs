//! # Solana MEV Bot
//!
//! Este bot está diseñado para aprovechar las oportunidades de MEV (Maximal Extractable Value)
//! en la red de Solana. El bot utiliza diversas estrategias, como sniping y copy trading,
//! para identificar y ejecutar transacciones rentables.
//!
//! ## Características principales
//!
//! - Integración con múltiples DEX (Raydium, Serum, Orca) para obtener precios y ejecutar transacciones
//! - Estrategias especializadas de sniping y copy trading para maximizar las ganancias
//! - Optimización avanzada del rendimiento y escalabilidad
//! - Monitoreo en tiempo real y dashboard para rastrear el rendimiento y las métricas clave
//! - Cobertura exhaustiva de pruebas y documentación detallada
//!
//! ## Uso
//!
//! Para utilizar el bot de Solana MEV, siga estos pasos:
//!
//! 1. Configure las credenciales y los parámetros en el archivo `config.toml`
//! 2. Ejecute el bot con `cargo run --release`
//! 3. Monitoree el rendimiento y las métricas a través del dashboard en tiempo real
//! 4. Ajuste las estrategias y los parámetros según sea necesario para optimizar las ganancias
//!
//! ## Contribución
//!
//! Si desea contribuir a este proyecto, siga estas pautas:
//!
//! 1. Abra un issue para discutir los cambios propuestos
//! 2. Haga fork del repositorio y cree una nueva rama para sus cambios
//! 3. Envíe un pull request con una descripción detallada de sus cambios y su propósito
//! 4. Asegúrese de que todos los tests pasen y de seguir las pautas de codificación establecidas
//!
//! ## Licencia
//!
//! Este proyecto está licenciado bajo los términos de la Licencia MIT. Consulte el archivo `LICENSE` para obtener más detalles.
//!

pub mod bot;
pub mod dex;
pub mod strategies;
pub mod models;
pub mod utils;