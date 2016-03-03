package main

import (
	"fmt"
	linear "github.com/ohheydom/linearregression"
)

func main() {
	// Generate Samples
	xTrain := [][]float64{[]float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2, 2}, []float64{3, 3, 3, 3, 3}, []float64{4, 4, 4, 4, 4}}
	yTrain := []float64{1, 2, 3, 4}
	xTest := [][]float64{[]float64{5, 5, 5, 5, 5}, []float64{6, 6, 6, 6, 6}, []float64{7, 7, 7, 7, 7}, []float64{8, 8, 8, 8, 8}}
	yTest := []float64{5, 6, 7, 8}

	// Fit
	lr := linear.LinearRegression{NIter: 500, Method: "gd"}
	lr.Fit(xTrain, yTrain)

	// Predict
	yPred := lr.Predict(xTest)

	// Calculate Error
	meanSquaredError := linear.MeanSquaredError(yTest, yPred)
	fmt.Println(meanSquaredError)
}
