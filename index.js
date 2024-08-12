require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCsv = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  );
}

/*
 MELHORES INDICADORES ATÉ AGORA:
 - lat
 - long
 - sqft_lot
 - sqft_living
 - yr_built
 - floors
 - bathrooms

 MELHOR RESULTADO - 9.5% de margem de erro média

*/

/* 
 Para encontrar o preço de casas, 9.5% de margem de erro ainda é muito alto.
 é um valor alto demais ainda se levarmos em conta os preços altos de 
 determiandas casas.
 Ex: Uma casa que vale 400.000 reais pode chegar a ter em média um erro de 38.000 reais
*/

/* 
 INDICADORES RUINS:
 - sqft_above
 - bedrooms
 - condition
 - grade
 - yr_renovated
 - sqft_basement

*/

let { features, labels, testFeatures, testLabels } = loadCsv(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot",'sqft_living', 'yr_built', 'floors', 'bathrooms'],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

let errorMean = 0;

testFeatures.forEach((testPoint, i) => {

  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = ((testLabels[i][0] - result) / testLabels[i][0]);

  // Caso a porcentagem dê negativa, significa que o resultado deduzido foi maior que o valor real
  // Caso a porcentagem dê positiva, significa que o resultado deduzido foi menor que o valor real

  errorMean += Math.abs(err); // Aqui estou somando as margens de erro.

  console.log(` ${i} - [ERROR]: ${(err * 100).toPrecision(2) }%`);
});

console.log(`Mean error: ${((errorMean / testFeatures.length) * 100).toPrecision(2)}%`);
