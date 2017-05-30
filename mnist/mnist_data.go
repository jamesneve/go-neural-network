package mnist

import (
	"github.com/petar/GoMNIST"
	"github.com/jamesneve/neuralnetwork2/learn"
)

type MnistData struct {
	trainingData *GoMNIST.Set
	testData *GoMNIST.Set
}

func NewMnistData() MnistData {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	return MnistData{
		trainingData: train,
		testData: test,
	}
}

func (m *MnistData) MakeTrainingData() []learn.TrainingData {
	return m.marshallData(m.trainingData)
}

func (m *MnistData) MakeTestData() []learn.TrainingData {
	return m.marshallData(m.testData)
}

func (m *MnistData) marshallData(dataset *GoMNIST.Set) []learn.TrainingData {
	result := make([]learn.TrainingData, 0, dataset.Count())
	for i := 0; i < dataset.Count(); i++ {
		image, label := dataset.Get(i)

		vectorizedImage := make([]float64, 0, 784)
		bounds := image.Bounds()
		for x := 0; x < bounds.Max.X; x++ {
			for y := 0; y < bounds.Max.Y; y++ {
				r, _, _, _ := image.At(x, y).RGBA()
				vectorizedImage = append(vectorizedImage, float64(r)/100000)
			}
		}
		t := learn.TrainingData{
			TrainingInput: vectorizedImage,
			DesiredOutputs: m.vectorizeOutput(uint8(label)),
		}
		result = append(result, t)
	}

	return result
}

func (m *MnistData) vectorizeOutput(n uint8) []float64 {
	res := make([]float64, 10)
	res[n] = 1.0
	return res
}
