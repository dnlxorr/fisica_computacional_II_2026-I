mod descomposicion_lu;
mod autovalores_autovectores;

use ndarray::{array, Array1, Array2, s};
use std::time::{Duration, Instant};
use crate::autovalores_autovectores::{autovectores_autovalores, matrices_qr};

fn main() {

    let A = array![
        [1.0, 4.0, 8.0, 4.0],
        [4.0, 2.0, 3.0, 7.0],
        [8.0, 3.0, 6.0, 9.0],
        [4.0, 7.0, 9.0, 2.0]
    ];

    let (Q, R) = matrices_qr(&A);

    println!("Matriz Q:\n{:?}", Q);
    println!("Matriz R:\n{:?}", R);

    let n = Q.dot(&R);
    println!("Q * R:\n{:?}", n);

    let (autovalores, V, A_final) = autovectores_autovalores(A);

    println!("Autovalores:\n{:?}", autovalores);
    println!("Matriz A final:\n{:?}", A_final);
    println!("Autovectores (V):\n{:?}", V);
}

