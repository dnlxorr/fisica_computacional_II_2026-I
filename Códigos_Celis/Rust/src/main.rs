mod descomposicion_lu;
mod matrices_tridiagonales_banda;
mod gauss;

use ndarray::array;
fn main() {
    let (x) = matrices_tridiagonales_banda::ejercicio_masasresortes(26);
    println!("{:?}",x);

}

