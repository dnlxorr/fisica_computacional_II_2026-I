// Solucion de la EDO -u''(x) = -pi^2 * sin(pi*x) con SOR
// Condiciones de frontera: u(0) = 0, u(1) = 0
// Soluci0n exacta: u(x) = sin(pi*x)

use std::f64::consts::PI;

fn main() {
    // Número de puntos interiores
    let n: usize = 20;

    // Dominio
    let a = 0.0_f64;
    let b = 1.0_f64;

    // Paso
    let h = (b - a) / (n + 1) as f64;

    // Vector de puntos (incluyendo frontera)
    let x: Vec<f64> = (0..=n + 1).map(|i| a + i as f64 * h).collect();

    // Inicialización de la solución
    let mut u = vec![0.0_f64; n + 2];

    // Condiciones de frontera
    u[0] = 0.0;
    u[n + 1] = 0.0;

    // Lado derecho f(x)
    let f: Vec<f64> = x.iter().map(|&xi| -(PI * PI) * (PI * xi).sin()).collect();

    // Parámetro de relajación
    let w = 1.5_f64;

    // Tolerancia y máximo de iteraciones
    let tol = 1e-6_f64;
    let max_iter = 10000;

    // Iteración SOR
    for k in 0..max_iter {
        let mut error = 0.0_f64;

        // Recorrer nodos interiores
        for i in 1..=n {
            let u_old = u[i];

            // Fórmula SOR
            u[i] = (1.0 - w) * u[i] + (w / 2.0) * (u[i - 1] + u[i + 1] - h * h * f[i]);

            // Cálculo del error
            let diff = (u[i] - u_old).abs();
            if diff > error {
                error = diff;
            }
        }

        if error < tol {
            println!("Convergió en {} iteraciones", k);
            break;
        }
    }

    // Solución exacta
    let u_exact: Vec<f64> = x.iter().map(|&xi| (PI * xi).sin()).collect();

    // Imprimir resultados (en lugar de graficar)
    println!("{:<10} {:<20} {:<20} {:<20}", "i", "x", "u_SOR", "u_exacta");
    println!("{}", "-".repeat(70));
    for i in 0..=n + 1 {
        println!("{:<10} {:<20.6} {:<20.6} {:<20.6}", i, x[i], u[i], u_exact[i]);
    }
}
