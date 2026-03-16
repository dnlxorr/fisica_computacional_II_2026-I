mod gauss;
mod descomposicion_lu;
use ndarray::{array, Array1, Array2, s};
use std::time::{Duration, Instant};
fn main() {
    let A = array![
        [4.0,-2.0,1.0],
        [20.0,-7.0,12.0],
        [-8.0,13.0,17.0]
    ];

    let (L_matrices, U, L_inv) = descomposicion_lu::descomposicion_lu(A.clone());

    for (i, L) in L_matrices.iter().enumerate() {
        println!("L {}", i);
        println!("{:?}", L);
    }

    println!("\nMatriz U:");
    println!("{:?}", U);

    println!("\nMatriz L inv:");
    println!("{:?}", L_inv);

    let A_rec = L_inv.dot(&U);

    println!("\n{:?}", A_rec);

}

