fn f(x: f64) -> f64 {
    (1.0 - x.ln()).sqrt()
    // 2.0 - (-x).exp()
    // (1.0 - x * x).exp()
    // (x - x * x).asin() * 0.5
}

fn derivada(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn metodo_relajacion(
    f: fn(f64) -> f64,
    xini: f64,
    tolerancia: f64,
    max_iter: usize,
) -> Option<(f64, usize)> {
    let mut x = xini;

    for i in 0..max_iter {
        let x_new = f(x);
        println!("Iter {}: x = {}", i, x_new);

        let d = derivada(f, x, 1e-5);

        if d.abs() >= 1.0 {
            println!("Error: El metodo puede no converger (|f'(x)| >= 1).");
            return None;
        }

        if (x_new - x).abs() < tolerancia {
            return Some((x_new, i));
        }

        x = x_new;
    }

    println!("Error: No convergió en el número máximo de iteraciones.");
    None
}

fn main() {
    let resultado = metodo_relajacion(f, 1.0, 1e-6, 100);

    if let Some((solucion, iteraciones)) = resultado {
        println!("Solución: {}", solucion);
        println!("Iteraciones: {}", iteraciones);
    }
}