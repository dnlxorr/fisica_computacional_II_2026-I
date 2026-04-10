use std::time::Instant;

fn main() {

    // Matriz A
    let a: Vec<Vec<f64>> = vec![
        vec![2.0, -1.0, 1.0],
        vec![3.0, 3.0, 9.0],
        vec![3.0, 3.0, 5.0],
    ];

    // Vector b
    let b: Vec<f64> = vec![2.0, -1.0, 4.0];

    let n = a.len();

    let inicio = Instant::now();

    // Matrices L y U
    let mut l = vec![vec![0.0; n]; n];
    let mut u = vec![vec![0.0; n]; n];

    // Descomposición LU
    for i in 0..n {

        // Calcular U
        for j in i..n {

            let mut suma = 0.0;

            for k in 0..i {
                suma += l[i][k] * u[k][j];
            }

            u[i][j] = a[i][j] - suma;
        }

        // Calcular L
        for j in i..n {

            if i == j {
                l[i][i] = 1.0;
            } else {

                let mut suma = 0.0;

                for k in 0..i {
                    suma += l[j][k] * u[k][i];
                }

                l[j][i] = (a[j][i] - suma) / u[i][i];
            }
        }
    }

    // Sustitución hacia adelante
    // Ly = b

    let mut y = vec![0.0; n];

    for i in 0..n {

        let mut suma = 0.0;

        for j in 0..i {
            suma += l[i][j] * y[j];
        }

        y[i] = b[i] - suma;
    }

    // Sustitución hacia atrás
    // Ux = y

    let mut x = vec![0.0; n];

    for i in (0..n).rev() {

        let mut suma = 0.0;

        for j in (i + 1)..n {
            suma += u[i][j] * x[j];
        }

        x[i] = (y[i] - suma) / u[i][i];
    }

    // Resultados

    println!("Matriz L:");
    for fila in &l {
        println!("{:?}", fila);
    }

    println!("\nMatriz U:");
    for fila in &u {
        println!("{:?}", fila);
    }

    println!("\nVector y:");
    println!("{:?}", y);

    println!("\nSolucion sistema x:");
    println!("{:?}", x);

    let duracion = inicio.elapsed();

    println!("\nTiempo (microsegundos):");
    println!("{}", duracion.as_micros());
}