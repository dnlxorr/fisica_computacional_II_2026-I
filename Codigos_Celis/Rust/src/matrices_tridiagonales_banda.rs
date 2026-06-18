use ndarray::{s, Array1, Array2};
use crate::descomposicion_lu;
use crate::gauss;

pub fn ejercicio_masasresortes(n: usize) -> Array1<f64>{
    const C: f64 = 1.0;
    const M: f64 = 1.0;
    const K: f64 = 6.0;
    const OMEGA: f64 = 2.0;

    const ALPHA: f64 = 2.0*K-M*OMEGA*OMEGA;

    // Creación de la matriz A

    let mut a  = Array2::<f64>::zeros((n,n));

    for i in 0..(n-1) {
        a[[i,i]] = ALPHA;
        a[[i,i+1]]=-K;
        a[[i+1,i]] = -K;
    }

    a[[0,0]] = ALPHA-K;
    a[[n-1,n-1]] = ALPHA-K;

    let mut v = Array1::<f64>::zeros((n));

    v[[0]] = ALPHA;

    let x = gauss::gauss(a,v);

    (x)
}
