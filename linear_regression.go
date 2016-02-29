package linearregression

// LinearRegression stores the parameters for creating a linear regression model
type LinearRegression struct {
	Weights []float64
	NIter   int
	method  string
}

// Fit will create the model and store the appropriate weights into the Weight field.
func (l *LinearRegression) Fit(x [][]float64, y []float64) {
	var nIter int
	if l.NIter == 0 {
		nIter = 200
	} else {
		nIter = l.NIter
	}

	if l.method == "gd" || l.method == "" {
		l.Weights = gdSolver(x, y, nIter, 0.02)
	}
}

// gdSolver returns the weights using a Gradient Descent algorithm.
//nIter is the number of iterations to run through before convergence, gamma is the step size.
func gdSolver(x [][]float64, y []float64, nIter int, gamma float64) []float64 {
	n := len(x)
	w := make([]float64, len(x[0])+1)
	for i := 0; i < nIter; i++ {
		predY := predY(x, w)
		errors := make([]float64, n)
		errorSum := 0.0
		for j := 0; j < n; j++ {
			errors[j] = y[j] - predY[j]
			errorSum += errors[j]
		}
		for k := 0; k < n; k++ {
			for l := 1; l < len(w); l++ {
				w[l] += gamma * x[k][l-1] * errors[k]
			}
		}
		w[0] += gamma * errorSum
	}
	return w
}

// predY uses the given weights to calculate each sample's label.
func predY(x [][]float64, w []float64) []float64 {
	n := len(x)
	nFeatures := len(x[0])
	predY := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 1; j <= nFeatures; j++ {
			predY[i] += x[i][j-1] * w[j]
		}
		predY[i] += w[0]
	}
	return predY
}
