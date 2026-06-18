use ndarray::{array, Array2};

pub fn descomposicion_lu(A:Array2<f64>) -> (Vec<Array2<f64>>, Array2<f64>, Array2<f64>) {
    let n = A.nrows();
    let mut matrices: Vec<Array2<f64>> = Vec::new();
    let mut matrices2: Vec<Array2<f64>> = Vec::new();

    let mut A_actual = A.clone();

    for c in 0..n{
        let mut L = Array2::<f64>::zeros((n,n));
        let mut L_inversa = Array2::<f64>::zeros((n,n));

        let pivote_actual = A_actual[[c,c]];

        for i in 0..n{
            L[[i,i]] = 1.0;
            L_inversa[[i,i]] = 1.0
        }

        for j in (c+1)..n{
            L[[j,c]] = -A_actual[[j,c]]/pivote_actual;
            L_inversa[[j,c]] = A_actual[[j,c]]/pivote_actual;
        }

        matrices.push(L.clone());
        matrices2.push(L_inversa.clone());

        A_actual = L.dot(&A_actual);
    }
    
    let mut L_total = Array2::<f64>::zeros((n,n));
    
    for i in 0..n{
        L_total[[i,i]] = 1.0;
    }
    
    for i in 0..matrices2.len(){
        L_total = L_total.dot(&matrices2[i]);
    }

    (matrices,A_actual,L_total)


}
