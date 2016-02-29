package main

import (
	"fmt"
	"github.com/ohheydom/linearregression"
)

func main() {
	x := [][]float64{[]float64{1, 1}, []float64{2, 2}, []float64{3, 3}, []float64{4, 4}}
	y := []float64{1, 2, 3, 4}
	lr := linearregression.LinearRegression{NIter: 800}
	lr.Fit(x, y)
	fmt.Println(lr.Weights)
}
