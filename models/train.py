import data

tNumFields = len(data.EURUSD15[0])
tNumTotalData = len(data.EURUSD15)

[tDateRaw, tTimeRaw, tOpen, tHigh, tLow, tClose, tVolume] = [[datum[i] for datum in data.EURUSD15] for i in range(tNumFields)]

tDate = [[int(tField) for tField in tDatum.split("-")] for tDatum in tDateRaw]
tTime = [[int(tField) for tField in tDatum.split(":")] for tDatum in tTimeRaw]

import random
import statistics

# set a random seed.  This makes below repeatable.  comment this out for a different result each time.
random.seed(10)

tNumFields = 4 # open, high, low, close

# this is the number of time steps in the observation period
tLengthX = 200

# this is the number of time steps in the prediction period
tLengthY = 20

# this is the number of observation periods to be sampled from the training data
tNumSamples = 1000

# fraction of data to be used for training
tTrainingFraction = 0.8

# number of neurons in hidden layer 1
tNumLayer1 = 800
# number of neurons in hidden layer 2
tNumLayer2 = 400

# batch size
tBatchSize = 100

# number of epochs
tNumEpochs = 20

# number of currency to buy
tPurchaseSize = 1000

# minimum margin for purchase
tMinMargin = 0.005

# maximum loss
tMaxLoss = 0.005

tNumTrainData = int(tTrainingFraction*tNumTotalData)
tNumTestData = tNumTotalData-tNumTrainData

# generate the random offsets into the training data
tRTrain = [random.randint(0,tNumTrainData-tLengthX-tLengthY) for i in range(tNumSamples)]
# generate the X training data [tNumSamples][tLengthX][tNumFields]
X_train = [[[tOpen[tRTrain[j]+i], tHigh[tRTrain[j]+i], tLow[tRTrain[j]+i], tClose[tRTrain[j]+i]] for i in range(tLengthX)] for j in range(tNumSamples)]
# generate the Y training data [tNumSamples][average, min, max]
Y_train = [[statistics.mean(tClose[tRTrain[j]+tLengthX:tRTrain[j]+tLengthX+tLengthY]),\
            min(tClose[tRTrain[j]+tLengthX:tRTrain[j]+tLengthX+tLengthY]),\
            max(tClose[tRTrain[j]+tLengthX:tRTrain[j]+tLengthX+tLengthY])\
           ] for j in range(tNumSamples)]

# generate the random offsets into the test data
tRTest = [random.randint(tNumTrainData,tNumTotalData-tLengthX-tLengthY) for i in range(tNumSamples)]
# generate the X test data [tNumSamples][tLengthX][tNumFields]
X_test = [[[tOpen[tRTest[j]+i], tHigh[tRTest[j]+i], tLow[tRTest[j]+i], tClose[tRTest[j]+i]] for i in range(tLengthX)] for j in range(tNumSamples)]
# generate the Y data [tNumSamples][tLengthY][tNumFields]
Y_data = [[[tOpen[tRTest[j]+tLengthX+i], tHigh[tRTest[j]+tLengthX+i], tLow[tRTest[j]+tLengthX+i], tClose[tRTest[j]+tLengthX+i]] for i in range(tLengthY)] for j in range(tNumSamples)]
# generate the Y test data [tNumSamples][average, min, max]
Y_test = [[statistics.mean(tClose[tRTest[j]+tLengthX:tRTest[j]+tLengthX+tLengthY]),\
           min(tClose[tRTest[j]+tLengthX:tRTest[j]+tLengthX+tLengthY]),\
           max(tClose[tRTest[j]+tLengthX:tRTest[j]+tLengthX+tLengthY]),\
          ] for j in range(tNumSamples)]

from tensorflow.keras import layers, models, optimizers, utils, datasets

input_layer = layers.Input((tLengthX,tNumFields))
x = layers.Flatten()(input_layer)
x = layers.Dense(tNumLayer1, activation="relu")(x)
x = layers.Dense(tNumLayer2, activation="relu")(x)

output_layer = layers.Dense(len(Y_train[0]))(x)

model = models.Model(input_layer, output_layer)

model.summary()

opt = optimizers.Adam()
model.compile(loss="mean_squared_error", optimizer=opt)

model.fit(X_train, Y_train, batch_size=tBatchSize, epochs=tNumEpochs)

model.evaluate(X_test, Y_test)

preds = model.predict(X_test)

change = 0.0
for index,datum in enumerate(X_test):
  # The close price of the observation period is the purchase price
  tPurchasePrice = datum[-1][-1]
  # If the price is predicted to go two std devs higher, then buy
  tPredictedMean = preds[index][0]
  if tPurchasePrice+tMinMargin > tPredictedMean:
    print(f'buy: {tPurchasePrice}, sell: {tPurchasePrice+tMinMargin}, pred mean: {tPredictedMean}.  No buy.')
    continue
  # Sell when the price reaches the expected average
  tSellTargetPrice = preds[index][0]
  # loop through the hold period
  tSoldAt = None
  for indexY,datumY in enumerate(Y_data[index]):
    # if the price reaches the target
    if datumY[3] >= tSellTargetPrice or datumY[3] <= tPurchasePrice-tMaxLoss:
      tSoldAt = datumY[3]
      break
  if tSoldAt == None:
    print('not reached')
    tSoldAt = Y_data[index][-1][3]
  tYield = tPurchaseSize*(tSoldAt-tPurchasePrice)
  print(f'bought: {tPurchasePrice}, sold: {tSoldAt}, yield: {tYield}')
  change += tYield
print(f'change: {change}')
