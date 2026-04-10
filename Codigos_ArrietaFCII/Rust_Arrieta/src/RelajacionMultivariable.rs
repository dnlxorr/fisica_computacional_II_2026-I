fn g(x: &[f64; 3]) -> [f64; 3] {
    [
        (x[1].powi(2) + x[2] + 1.0) / 3.0,
        (x[0] + x[2].powi(2) + 1.0) / 3.0,
        (x[0] + x[1] + 1.0) / 3.0,
    ]
    // [(( x[1] + 5.0) / 2.0).sqrt(),
    //  (x[0] + 1.0).sqrt(),
    //  0.0]
}

fn norma(v: &[f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn relajacion(
    g: fn(&[f64; 3]) -> [f64; 3],
    x0: [f64; 3],
    lam: f64,
    tol: f64,
    max_iter: usize,
) -> ([f64; 3], usize) {
    let mut x = x0;

    for i in 0..max_iter {
        let gx = g(&x);
        let x_new = [
            x[0] + lam * (gx[0] - x[0]),
            x[1] + lam * (gx[1] - x[1]),
            x[2] + lam * (gx[2] - x[2]),
        ];

        println!("Iter {}: x = {:.6}, y = {:.6}, z = {:.6}", i, x_new[0], x_new[1], x_new[2]);

        let diff = [x_new[0] - x[0], x_new[1] - x[1], x_new[2] - x[2]];
        if norma(&diff) < tol {
            println!("convergio en {} iteraciones", i);
            return (x_new, i);
        }

        x = x_new;
    }

    println!("no convergio");
    (x, max_iter)
}

fn main() {
    let x0 = [0.3, 0.3, 0.3];

    let (solucion, it) = relajacion(g, x0, 0.5, 1e-8, 1000);

    println!(
        "Solución ≈ (x = {:.6}, y = {:.6}, z = {:.6}) en {} iteraciones",
        solucion[0], solucion[1], solucion[2], it
    );
}