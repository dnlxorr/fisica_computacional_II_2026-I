


pub fn main() {

    // Matriz A
    let mut a: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0, 4.0, 1.0],
        vec![3.0, 4.0, -1.0, -1.0],
        vec![1.0, -4.0, 1.0, 5.0],
        vec![2.0, -2.0, 1.0, 3.0],
    ];

    // Vector v
    let mut v: Vec<f64> = vec![-4.0, 3.0, 9.0, 7.0];

    let n = v.len();

    // Eliminación Gaussiana


    for fila in 0..n {

        // Partial pivoting
        let mut elem_max = fila;

        for i in (fila + 1)..n {
            if a[i][fila].abs() > a[elem_max][fila].abs() {
                elem_max = i;
            }
        }

        // Intercambiar filas
        if elem_max != fila {
            a.swap(fila, elem_max);
            v.swap(fila, elem_max);
        }

        // Dividir por el elemento diagonal
        let div = a[fila][fila];

        for j in 0..n {
            a[fila][j] /= div;
        }

        v[fila] /= div;

        // Eliminar filas inferiores
        for fila_inf in (fila + 1)..n {

            let mult = a[fila_inf][fila];

            for j in 0..n {
                a[fila_inf][j] -= mult * a[fila][j];
            }

            v[fila_inf] -= mult * v[fila];
        }
    }

    println!("Matriz triangular:");
    for fila in &a {
        println!("{:?}", fila);
    }


    // Back Substitution


    let mut x: Vec<f64> = vec![0.0; n];

    for fila in (0..n).rev() {

        x[fila] = v[fila];

        for i in (fila + 1)..n {
            x[fila] -= a[fila][i] * x[i];
        }
    }

    println!("Orden de filas:");
    println!("{:?}", (0..n).rev().collect::<Vec<_>>());

    println!("Solución:");
    println!("{:?}", x);
}