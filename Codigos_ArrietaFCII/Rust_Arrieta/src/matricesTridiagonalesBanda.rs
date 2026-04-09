use std::thread::park;
use ndarray::{array, array1, array2};
use crate::eliminacionGaussianaeInversa;
pub fn masaresorte(n:usize)->array2<f64>{

    const C: f64=1.0;
    const N: f64=26.0;
    const M: f64=1.0;
    const K: f64= 6.0;
    const omega: f64=  2.0;

    const alpha: f64= 2.0*K-M*omega*omega;

    let mut a=array2::<f64>::zeros(shape: (n,n));
    let mut a=array2::<f64>::zeros((n));


    let x = eliminacionGaussianaeInversa::main()








}

