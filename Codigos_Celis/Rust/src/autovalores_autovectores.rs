use ndarray::{Array2, Array1};
use ndarray::prelude::*;

// =======================
// Producto punto
// =======================
pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// =======================
// Norma vector
// =======================
pub fn norm(v: &Array1<f64>) -> f64 {
    dot(v, v).sqrt()
}

// =======================
// QR por Gram-Schmidt
// =======================
pub fn matrices_qr(A: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = A.nrows();

    let mut R = Array2::<f64>::zeros((n, n));
    let mut Q = Array2::<f64>::zeros((n, n));
    let mut u_normas = vec![0.0; n];

    for i in 0..n {
        let mut u = A.column(i).to_owned();

        for j in 0..i {
            let qj = Q.column(j);
            let proy = dot(&qj.to_owned(), &A.column(i).to_owned());

            u = &u - &(proy * &qj);
        }

        let norma = norm(&u);
        u_normas[i] = norma;

        let qi = u.mapv(|x| x / norma);
        Q.column_mut(i).assign(&qi);
    }

    for i in 0..n {
        R[[i, i]] = u_normas[i];

        for j in (i + 1)..n {
            let val = dot(&Q.column(i).to_owned(), &A.column(j).to_owned());
            R[[i, j]] = val;
        }
    }

    (Q, R)
}

// =======================
// Autovalores y autovectores (QR iterativo)
// =======================
pub fn autovectores_autovalores(mut A: Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let epsilon = 1e-10;
    let n_iter = 5000;
    let n = A.nrows();

    let mut V = Array2::<f64>::eye(n);

    for _ in 0..n_iter {
        let (Q, R) = matrices_qr(&A);

        A = R.dot(&Q);
        V = V.dot(&Q);

        let mut criterio = true;

        for i in 0..n {
            for j in 0..n {
                if i != j && A[[i, j]].abs() > epsilon {
                    criterio = false;
                    break;
                }
            }
            if !criterio {
                break;
            }
        }

        if criterio {
            break;
        }
    }

    let autovalores = A.diag().to_owned();

    (autovalores, V, A)
}

