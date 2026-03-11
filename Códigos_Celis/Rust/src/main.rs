use ndarray::{array,Array1,Array2,s};
use std::time::{Duration, Instant};
fn main() {
    let mut a: Array2<f64> = array![
        [3.0,4.0,5.0,2.0,3.0],
        [2.0,5.0,10.0,11.7,2.0],
        [-4.0,4.0,-3.0,3.9,3.0],
        [2.0,6.0,-7.0,9.0,10.0],
        [3.0,5.0,7.0,8.0,5.0]
    ];

    let mut v: Array1<f64> = array![3.0,8.0,5.0,3.5,2.0];

    let n = v.len();

    // Eliminción Gaussiana
    let start=Instant::now();
    for fila in 0..n{
        let div = a[[fila,fila]];

        {
            let mut row= a.slice_mut(s![fila,..]);
            row /= div;
        }

        v[fila] /= div;

        for fila_inf in (fila+1)..n{
            let mult = a[[fila_inf,fila]];

            let fila_pivote = a.row(fila).to_owned();
            let mut fila_obj = a.row_mut(fila_inf);

            fila_obj -= &(fila_pivote*mult);

            v[fila_inf] -= mult*v[fila];
        }

    }

    println!("Matriz triangular:");
    println!("{:?}",a);

    // Backsustitution

    // MAnera de crear una matriz vacía
    let mut x = Array1::<f64>::zeros(n);

    for fila in (0..n).rev(){
        x[fila] = v[fila];

        for i in (fila+1)..n{
            x[fila] -= a[[fila,i]]*x[i];
        }
    }

    let duration=start.elapsed();
    println!("Orden de iteración");
    println!("{:?}",(0..n).rev().collect::<Vec<_>>());

    print!("Solución:");
    println!("{:?}",x);
    println!("{:?}",duration);
}

