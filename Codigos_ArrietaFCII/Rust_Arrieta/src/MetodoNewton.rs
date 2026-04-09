fn funcion(x: f64) -> f64 {
    x.powi(3) - x - 2.0
    // x.cos() - x
}

fn derivada_funcion(x: f64) -> f64 {
    3.0 * x.powi(2) - 1.0
    // -x.sin() - 1.0
}

fn metodo_newton(
    funcion: fn(f64) -> f64,
    derivada_funcion: fn(f64) -> f64,
    x0: f64,
    tolerancia: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut x = x0;

    for i in 0..max_iter {
        let fx = funcion(x);
        let dfx = derivada_funcion(x);

        if dfx.abs() < 1e-12 {
            println!("Error: la derivada de la funcion es 0 en el punto. El metodo no puede continuar");
            return None;
        }

        let x_new = x - fx / dfx;
        let error = (x_new - x).abs();

        println!(
            "Iteracion {}: x = {:.6},  f(x) = {:.2},  error = {:.2e}",
            i + 1, x_new, funcion(x_new), error
        );

        if error < tolerancia {
            println!("solucion convergente, la raiz aproximada es: {:.6}", x_new);
            return Some(x_new);
        }

        x = x_new;
    }

    println!("NO convergió en el rango de iteraciones dada");
    None
}

fn main() {
    println!("f(x) = x³ - x - 2\n");
    let _raiz = metodo_newton(funcion, derivada_funcion, 2.0, 1e-8, 100);

    // println!("\n>>> f(x) = cos(x) - x\n");
    // let _raiz = metodo_newton(funcion, derivada_funcion, 1.0, 1e-8, 100);
}