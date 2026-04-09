


fn main() {
 
    let a: Vec<Vec<f64>> = vec![
        vec![2.0, 1.0, 1.0],
        vec![1.0, 3.0, 2.0],
        vec![1.0, 0.0, 0.0],
    ];


    match matriz_inversa(&a) {
        Ok(a_inv) => {
            println!("Matriz A:");
            imprimir_matriz(&a);

            println!("\nInversa de A:");
            imprimir_matriz(&a_inv);

            println!("\nVerificación A * A_inv = I:");
            let verificacion = multiplicar_matrices(&a, &a_inv);
            imprimir_matriz(&verificacion);
        }
        Err(e) => println!("Error: {}", e),
    }
}


fn matriz_inversa(a: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();

    
    let identidad = crear_identidad(n);

   
    let mut a_inv = vec![vec![0.0_f64; n]; n];

   
    for i in 0..n {
       
        let e: Vec<f64> = identidad.iter().map(|fila| fila[i]).collect();

       
        let x = resolver_sistema(a, &e)?;  

       
       
        for j in 0..n {
            a_inv[j][i] = x[j];
        }
    }

    Ok(a_inv)
}


fn resolver_sistema(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Result<Vec<f64>, String> {
    let n = a.len();

    
    let mut aumentada: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, fila)| {
            let mut nueva_fila = fila.clone();
            nueva_fila.push(b[i]);  
            nueva_fila
        })
        .collect();

    
    for col in 0..n {
        // Pivoteo parcial: buscamos la fila con el mayor valor en la columna
        let fila_pivot = (col..n)
            .max_by(|&i, &j| {
                aumentada[i][col]
                    .abs()
                    .partial_cmp(&aumentada[j][col].abs())
                    .unwrap()
            })
            .unwrap();

        // Si el pivote es casi cero, la matriz es singular
        if aumentada[fila_pivot][col].abs() < 1e-12 {
            return Err("La matriz es singular o casi singular".to_string());
        }

        // Intercambiamos la fila actual con la del pivote
        aumentada.swap(col, fila_pivot);

        // Eliminamos hacia abajo
        for fila in (col + 1)..n {
            let factor = aumentada[fila][col] / aumentada[col][col];
            for j in col..=n {
                let val = aumentada[col][j] * factor;
                aumentada[fila][j] -= val;
            }
        }
    }

    // --- SUSTITUCIÓN HACIA ATRÁS ---
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = aumentada[i][n];
        for j in (i + 1)..n {
            let val = aumentada[i][j] * x[j];
            x[i] -= val;
        }
        x[i] /= aumentada[i][i];
    }

    Ok(x)
}


fn crear_identidad(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}


fn multiplicar_matrices(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}


fn imprimir_matriz(m: &Vec<Vec<f64>>) {
    for fila in m {
        let valores: Vec<String> = fila.iter().map(|v| format!("{:8.4}", v)).collect();
        println!("[{}]", valores.join(", "));
    }
}