use std::time::Instant;
use std::thread;

fn main() {
    let start = Instant::now();

    // Código a medir
    let mut sum = 0;
    let handle = thread::spawn(|| {
    for i in 0..100000 {
        sum += i;
    }

    let duration = start.elapsed();
    println!("Tiempo de ejecución: {:?}", duration);
    // Para verlo en milisegundos:
    println!("Milisegundos: {}ms", duration.as_millis());
    });
}
