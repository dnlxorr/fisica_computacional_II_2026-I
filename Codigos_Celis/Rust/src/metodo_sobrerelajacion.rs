// Sistema Ax = b resuelto con el metodo de sobrerelajacion (SOR)

fn main() {
    let a = [
        [4.0_f64, -1.0, -6.0,  0.0],
        [-5.0,    -4.0, 10.0,  8.0],
        [0.0,      9.0,  4.0, -2.0],
        [1.0,      0.0, -7.0,  5.0],
    ];

    let b = [2.0_f64, 21.0, -12.0, -6.0];

    let omega = 0.5_f64;
    let n = b.len();
    let mut x = vec![0.0_f64; n];

    let tolerancia = 1e-8_f64;
    let mut error = 1.0_f64;

    while error > tolerancia {
        let x_antigua = x.clone();

        for i in 0..n {
            let suma1: f64 = (0..i).map(|j| a[i][j] * x[j]).sum();
            let suma2: f64 = (i + 1..n).map(|j| a[i][j] * x_antigua[j]).sum();

            x[i] = (1.0 - omega) * x_antigua[i]
                + (omega / a[i][i]) * (b[i] - suma1 - suma2);

            println!("{}", x[i]);
        }

        error = x.iter()
            .zip(x_antigua.iter())
            .map(|(xi, xi_old)| (xi - xi_old).powi(2))
            .sum::<f64>()
            .sqrt();
    }

    println!("Solución: {:?}", x);
}
