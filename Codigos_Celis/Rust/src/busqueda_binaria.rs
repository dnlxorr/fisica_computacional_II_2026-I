// Resolver f(x) = x^3 - x - 2 usando bisección

fn f(x: f64) -> f64 {
    x.powi(3) - x - 2.0
}

fn main() {
    let mut a = 1.0_f64;
    let mut b = 2.0_f64;
    let tolerancia = 1e-6_f64;
    let mut error = 1.0_f64;
    let mut iteracion = 0u32;
    let mut c = 0.0_f64;

    while error > tolerancia {
        c = (a + b) / 2.0;

        let fa = f(a);
        let fc = f(c);

        println!("Iteración: {}", iteracion);
        println!("a = {} b = {} c = {}", a, b, c);
        println!("f(c) = {}", fc);
        println!("-----------------------");

        if fa * fc < 0.0 {
            b = c;
        } else {
            a = c;
        }

        error = (b - a).abs();
        iteracion += 1;
    }

    println!("Raíz aproximada: {}", c);
}
