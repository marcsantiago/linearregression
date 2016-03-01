package main

import (
	"fmt"
	linear "github.com/ohheydom/linearregression"
)

func main() {
	x := [][]float64{[]float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2, 2}, []float64{3, 3, 3, 3, 3}, []float64{4, 4, 4, 4, 4}}
	y := []float64{1, 2, 3, 4}
	lr := linear.LinearRegression{NIter: 200, Method: "gd"}
	lr.Fit(x, y)
	fmt.Println(lr.Weights)
}