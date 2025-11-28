# Neural Network Roadmap - MNIST Digit Recognition

Dit document beschrijft de voortgang en planning voor het bouwen van een neural network from scratch in C++ voor het herkennen van handgeschreven cijfers (MNIST dataset).

## Doel
Een werkend neural network bouwen dat de MNIST digits (0-9) kan herkennen, volledig from scratch geïmplementeerd in C++ als leerproject.

---

## ✅ Fase 1: Basis Infrastructure (AFGEROND)

### Linear Algebra
- [x] Matrix class met basis operaties
  - [x] Matrix multiplicatie (matrix * matrix)
  - [x] Matrix-vector multiplicatie
  - [x] Matrix optelling
  - [x] Transpose functie
  - [x] Hadamard product
- [x] Vector operaties (dot product, add_vectors)
- [x] Random weight initialisatie (utils.h)

### Neural Network Basis
- [x] Layer class met weights en biases
- [x] Activation functions implementeren
  - [x] ReLU
  - [x] Softmax
- [x] Model class met configureerbare dimensies
- [x] Forward pass implementatie
- [x] Loss function: Categorical Cross-Entropy
- [x] Loss gradient berekening

### Model Persistence
- [x] Model save functionaliteit (.jesperstijn format)
- [x] Model load functionaliteit
- [x] Basis test file voor forward pass

---

## 🔄 Fase 2: Training Capability (IN PROGRESS)

### 2.1 Backpropagation - Basis Begrip

**Concept:** Backpropagation berekent hoe elk gewicht en bias de loss beïnvloedt, zodat we weten in welke richting we ze moeten aanpassen.

#### [ ] Stap 1: Activatie derivatives implementeren

**Wat:** De afgeleide van je activation functions.

**Waarom:** Bij backpropagation moet je de gradient "terug" propageren. De chain rule vereist dat je weet hoe de activation function verandert.

**ReLU Derivative:**
```cpp
// In Layer class (model.hpp/cpp)
static std::vector<double> relu_derivative(std::vector<double> vec);

// Implementatie:
// relu'(x) = 1 if x > 0, else 0
// Je hebt de ORIGINELE input nodig (voor de activatie werd toegepast)
```

**Softmax + Cross-Entropy Derivative (gecombineerd):**
```cpp
// Special case: softmax + categorical cross-entropy samen heeft een simpele gradient!
// gradient = prediction - target (dit heb je al in categorical_cross_entropy_loss_gradient!)
```

**Hint:** Bewaar de "pre-activation" waarden (voor ReLU wordt toegepast) in je Layer class tijdens forward pass. Je hebt ze nodig voor backward pass.

**Implementatie tips:**
- Voeg toe aan Layer class: `std::vector<double> z;` (pre-activation waarden)
- Update `activate()` functie om z op te slaan voor backprop

#### [ ] Stap 2: Layer activations opslaan tijdens forward pass

**Probleem:** Je huidige forward pass slaat activations NIET op in this->layers[i].

**Bug in huidige code (model.cpp:134):**
```cpp
Layer current_layer = this->layers[i];  // Dit is een KOPIE!
current_layer.activate(next);           // Werkt op de kopie
// De activation in this->layers[i] wordt NIET geüpdatet!
```

**Oplossing:**
```cpp
// Gebruik reference:
Layer& current_layer = this->layers[i];  // Let op de &
current_layer.activate(next);
// Of direct:
this->layers[i].activate(next);
```

**Waarom belangrijk voor backprop:** Je hebt alle layer activations nodig om gradients te berekenen.

#### [ ] Stap 3: Gradient storage toevoegen aan Layer

**Wat toevoegen aan Layer class:**
```cpp
// In model.hpp, Layer class:
std::vector<double> z;              // Pre-activation (weighted sum + bias)
std::vector<double> delta;          // Gradient van loss t.o.v. z
Matrix weight_gradients;            // Gradient van loss t.o.v. weights
std::vector<double> bias_gradients; // Gradient van loss t.o.v. biases
```

**Waarom:** Deze variabelen slaan tijdelijk de berekende gradients op tijdens één backward pass.

#### [ ] Stap 4: Backward pass voor één layer implementeren

**Concept:** Voor één layer, bereken hoe weights en biases aangepast moeten worden.

**Formules (voor layer i):**
```
delta[i] = gradient_from_next_layer * activation_derivative(z[i])
weight_gradients[i] = delta[i] * activation[i-1].transpose()
bias_gradients[i] = delta[i]
gradient_to_previous_layer = weights[i].transpose() * delta[i]
```

**Implementatie:**
```cpp
// In Layer class:
std::vector<double> backward(std::vector<double> gradient_from_next,
                              std::vector<double> prev_activation);
// Returns: gradient to pass to previous layer
```

**Stappen in de functie:**
1. Bereken delta (gradient * activation_derivative)
2. Bereken weight_gradients (delta als column vector × prev_activation als row vector)
3. Bereken bias_gradients (gewoon delta)
4. Bereken gradient voor vorige layer (weights.transposed() * delta)
5. Return gradient voor vorige layer

**Hint:** Voor de output layer is `gradient_from_next` de loss gradient (prediction - target).

#### [ ] Stap 5: Backward pass door hele netwerk

**Wat:** Een functie in Model class die backward pass door alle layers doet.

```cpp
// In Model class (model.hpp):
void backward(std::vector<double> target);

// Implementatie in model.cpp:
void Model::backward(std::vector<double> target) {
    // 1. Start met loss gradient (output layer)
    std::vector<double> gradient = categorical_cross_entropy_loss_gradient(
        layers.back().activation, target);

    // 2. Loop backwards door layers (van laatste naar eerste)
    for (int i = layers.size() - 1; i >= 1; i--) {
        // Get previous layer's activation
        std::vector<double> prev_activation = layers[i-1].activation;

        // Backward pass voor deze layer
        gradient = layers[i].backward(gradient, prev_activation);
    }
}
```

**Belangrijke details:**
- Loop van achter naar voren (i = layers.size()-1 tot i = 1)
- Skip input layer (i = 0) want die heeft geen weights
- Geef activation van vorige layer mee

#### [ ] Stap 6: Gradient checking implementeren (TESTING!)

**Wat:** Vergelijk je analytische gradients met numerieke gradients om bugs te vinden.

**Numerieke gradient formule:**
```
gradient ≈ (loss(weight + epsilon) - loss(weight - epsilon)) / (2 * epsilon)
```

**Implementatie:**
```cpp
// Test functie (bijv. in test.cpp of nieuwe test file):
void test_gradients() {
    // 1. Maak klein netwerk (bijv. 3 -> 4 -> 2)
    // 2. Doe forward + backward pass
    // 3. Voor een paar random weights:
    //    - Bereken numerieke gradient
    //    - Vergelijk met analytische gradient
    //    - Check dat verschil < 1e-5
}
```

**Hint:** Start met één weight testen, dan uitbreiden.

### 2.2 Optimizer - Gradient Descent

#### [ ] Stap 7: Learning rate parameter toevoegen

```cpp
// In Model class:
double learning_rate = 0.01;  // Default waarde

// Of in constructor:
Model(std::vector<int> dimensions,
      std::string activation_function,
      std::string loss_function,
      double learning_rate = 0.01);
```

#### [ ] Stap 8: Update weights functie implementeren

**Wat:** Pas alle weights en biases aan op basis van gradients.

```cpp
// In Model class:
void update_weights();

// Implementatie:
void Model::update_weights() {
    for (int i = 1; i < layers.size(); i++) {
        // Update weights: w = w - learning_rate * gradient
        // Update biases: b = b - learning_rate * gradient

        // LET OP: Je moet matrix en vector aftrekking implementeren!
        // Of doe het element-by-element in nested loops

        for (int row = 0; row < layers[i].weights.rows; row++) {
            for (int col = 0; col < layers[i].weights.columns; col++) {
                layers[i].weights[row][col] -=
                    learning_rate * layers[i].weight_gradients[row][col];
            }
        }

        for (int j = 0; j < layers[i].biases.size(); j++) {
            layers[i].biases[j] -= learning_rate * layers[i].bias_gradients[j];
        }
    }
}
```

#### [ ] Stap 9: Train step functie (één sample)

**Wat:** Combineer forward, backward, en update in één functie.

```cpp
// In Model class:
double train_step(std::vector<double> input, std::vector<double> target);

// Implementatie:
double Model::train_step(std::vector<double> input, std::vector<double> target) {
    // 1. Forward pass
    std::vector<double> prediction = forward(input);

    // 2. Calculate loss
    double loss = loss_function(prediction, target, 0.0000001);

    // 3. Backward pass
    backward(target);

    // 4. Update weights
    update_weights();

    return loss;
}
```

### 2.3 Training Loop

#### [ ] Stap 10: Batch training implementeren

**Concept:** Train op meerdere samples, gemiddelde de gradients.

**Simpele versie (eerst implementeren):**
```cpp
// Train op alle samples één voor één
void train_epoch(std::vector<std::vector<double>> inputs,
                 std::vector<std::vector<double>> targets) {
    double total_loss = 0;
    for (int i = 0; i < inputs.size(); i++) {
        total_loss += train_step(inputs[i], targets[i]);
    }
    std::cout << "Epoch loss: " << total_loss / inputs.size() << std::endl;
}
```

**Uitgebreide versie (later):**
- Gradient accumulation over meerdere samples
- Weight update pas na hele batch

#### [ ] Stap 11: Multi-epoch training

```cpp
void train(std::vector<std::vector<double>> inputs,
           std::vector<std::vector<double>> targets,
           int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        train_epoch(inputs, targets);
    }
}
```

#### [ ] Stap 12: Test met XOR of simple dataset

**Voor je MNIST doet:** Test met simpel probleem!

**XOR voorbeeld:**
```cpp
// Input: 4 samples van 2 features
std::vector<std::vector<double>> xor_inputs = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

// Targets: one-hot encoded (2 classes: 0 or 1)
std::vector<std::vector<double>> xor_targets = {
    {1, 0},  // 0 XOR 0 = 0
    {0, 1},  // 0 XOR 1 = 1
    {0, 1},  // 1 XOR 0 = 1
    {1, 0}   // 1 XOR 1 = 0
};

// Model: 2 -> 4 -> 2 (kleine hidden layer)
Model model({2, 4, 2}, "relu", "categorical_cross_entropy_loss", 0.1);
model.train(xor_inputs, xor_targets, 1000);
```

**Als XOR werkt,** dan werkt je backprop! Dan ben je klaar voor MNIST.

---

## 📊 Fase 3: MNIST Data Integration

### Data Loading
- [ ] MNIST dataset downloaden/integreren
- [ ] IDX file format parser (MNIST native format)
  - [ ] Image file reader (train-images-idx3-ubyte)
  - [ ] Label file reader (train-labels-idx1-ubyte)
- [ ] Data normalisatie (0-255 → 0-1 of -1 to 1)
- [ ] One-hot encoding voor labels (0-9 → vector van 10)

### Data Pipeline
- [ ] Training set loader (60,000 images)
- [ ] Test set loader (10,000 images)
- [ ] Batch generator
- [ ] (Optioneel) Data shuffling

---

## 🎯 Fase 4: Model Training & Evaluation

### Training Process
- [ ] Hyperparameters definiëren
  - [ ] Network architecture (bijv. 784 → 128 → 64 → 10)
  - [ ] Learning rate
  - [ ] Batch size
  - [ ] Number of epochs
- [ ] Training loop uitvoeren op MNIST
- [ ] Loss monitoring tijdens training
- [ ] Model checkpointing (save best model)

### Evaluation
- [ ] Accuracy metric implementeren
- [ ] Test set evaluatie
- [ ] Confusion matrix (optioneel)
- [ ] Per-digit accuracy analyse

### Testing
- [ ] Unit tests voor linalg operaties
- [ ] Gradient checking (numerical vs analytical)
- [ ] Overfitting check (train vs test accuracy)

---

## 🚀 Fase 5: Optimalisatie & Verbetering (TOEKOMST)

### Performance
- [ ] Code profiling
- [ ] Matrix operaties optimaliseren
- [ ] Mogelijk: parallelisatie (OpenMP)
- [ ] Memory management optimaliseren

### Features
- [ ] Learning rate scheduling/decay
- [ ] Early stopping
- [ ] Dropout layers (regularisatie)
- [ ] Batch normalization
- [ ] Different activation functions (tanh, sigmoid voor hidden layers)
- [ ] Different weight initialization strategies (Xavier, He)

### Data Augmentation
- [ ] Image rotatie
- [ ] Image translation
- [ ] Elastic deformations

### Visualization
- [ ] Training curve plots
- [ ] Learned weights visualisatie
- [ ] Misclassified examples analysis

---

## 📝 Huidige Prioriteit

**VOLGENDE STAP:** Start met Stap 1 - ReLU derivative implementeren

---

## 🐛 Gevonden Bugs & C++ Verbeteringen

### KRITIEKE BUGS (Fix deze eerst!)

#### 1. Layer copy bug in forward pass (model.cpp:125, 134)
```cpp
// FOUT:
Layer current_layer = this->layers[i];  // Maakt een KOPIE
current_layer.activate(next);           // Activation wordt niet opgeslagen!

// GOED:
Layer& current_layer = this->layers[i]; // Reference (let op &)
// Of:
this->layers[i].activate(next);
```

**Impact:** Je layers slaan hun activations NIET op, dus backward pass zal niet werken!

#### 2. Softmax max initialisatie bug (model.cpp:67)
```cpp
// FOUT:
double max = 0;  // Als alle waarden negatief zijn gaat dit mis!

// GOED:
double max = vec[0];  // Start met eerste element
// Of:
double max = -std::numeric_limits<double>::infinity();
```

#### 3. Matrix addition loop bug (linalg.cpp:77)
```cpp
// FOUT:
for (int j = 0; j < m2.rows; j++) {  // Moet columns zijn!

// GOED:
for (int j = 0; j < this->columns; j++) {
```

#### 4. Ongeïnitialiseerde variable (model.cpp:92)
```cpp
int prev_neuron_amount;  // NIET geïnitialiseerd!

// Fix: initialiseer altijd
int prev_neuron_amount = 0;
```

---

## 💡 C++ Tips voor Beginners

### 1. Pass by Reference voor grote objecten

**Probleem:** Je code kopieert veel vectors onnodig.

**Voorbeelden in je code:**
```cpp
// SLECHT (linalg.cpp:7, 19):
double dot_product(std::vector<double> v1, std::vector<double> v2)  // Kopieert!
std::vector<double> add_vectors(std::vector<double> v1, std::vector<double> v2)

// GOED:
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2)
std::vector<double> add_vectors(const std::vector<double>& v1, const std::vector<double>& v2)
```

**Regel:** Gebruik `const &` voor parameters die je alleen leest, niet wijzigt.

**Waarom:**
- Vector copy is duur (alloceert memory, kopieert alle elementen)
- `const &` = geen copy, alleen lezen toegestaan
- Voor kleine types (int, double) maakt het niet uit

**Waar in je code dit fixen:**
- `model.cpp:47` - `void Layer::activate(std::vector<double> activation_vector)`
- `model.cpp:56, 66` - ReLU en Softmax
- `model.cpp:85` - Model constructor (`std::string` parameters)
- `model.cpp:143, 154` - Loss functions
- `linalg.cpp:7, 19` - dot_product en add_vectors
- `linalg.cpp:41` - Matrix constructor loop: `for (const auto& row : data)`

### 2. size_t vs int voor vector indexing

```cpp
// SLECHT:
for (int i = 0; i < vec.size(); i++)  // vec.size() is size_t, niet int!

// GOED:
for (size_t i = 0; i < vec.size(); i++)
// Of moderne C++:
for (size_t i = 0; const auto& element : vec)
```

**Waarom:** `vec.size()` returned `size_t` (unsigned). Compiler warning bij vergelijken int met size_t.

**Je doet nu:** `(int) vec.size()` - werkt maar is hacky.

### 3. Range-based for loops

```cpp
// SLECHT (linalg.cpp:41):
for (std::vector<double> row: data)  // Kopieert elke row!

// GOED:
for (const auto& row : data)  // auto = compiler bepaalt type, & = reference

// Of als je wilt wijzigen:
for (auto& row : data)
```

**Voordelen:**
- Minder typewerk
- Geen index bugs
- Geen copy (met &)

### 4. Initialiseer altijd variabelen

```cpp
// GEVAARLIJK:
int x;           // Onbekende waarde!
double max;      // Kan alles zijn!

// VEILIG:
int x = 0;
double max = 0.0;
```

### 5. const correctness

```cpp
// Functies die niets wijzigen moeten const zijn:
class Layer {
    int get_neuron_count() const { return neuron_amount; }  // const!
};

// Parameters die je niet wijzigt:
void process(const std::vector<double>& data);
```

### 6. nullptr vs NULL vs 0

```cpp
// Ouderwets:
double* ptr = NULL;   // C-style
double* ptr = 0;      // Verwarrend

// Modern C++:
double* ptr = nullptr;
```

### 7. Vermijd naked pointers, gebruik references

```cpp
// Vermijd (tenzij nodig):
void process(Layer* layer)

// Gebruik:
void process(Layer& layer)  // Kan niet null zijn, veiliger
```

### 8. Member initializer lists (in constructors)

```cpp
// MINDER EFFICIENT:
Layer::Layer(int n, int prev_n, string act, bool first) {
    this->neuron_amount = n;  // Assignment
}

// BETER:
Layer::Layer(int n, int prev_n, string act, bool first)
    : neuron_amount(n),  // Direct initialisatie
      prev_neuron_amount(prev_n),
      first(first) {
    // Body
}
```

### 9. Gebruik emplace_back in plaats van push_back

```cpp
// GOED (je doet dit al!):
layers.emplace_back(dimensions[i], prev_neuron_amount, "softmax", first);

// MINDER EFFICIENT:
layers.push_back(Layer(dimensions[i], prev_neuron_amount, "softmax", first));
```

### 10. RAII en smart pointers (advanced, later)

Als je dynamische memory nodig hebt:
```cpp
// VERMIJD:
double* data = new double[100];
// ... vergeet delete[] = memory leak!

// GEBRUIK (C++11+):
std::vector<double> data(100);  // Automatic cleanup
// Of:
std::unique_ptr<double[]> data(new double[100]);
```

---

## 🔍 Code Review Checklist

Voordat je nieuwe code commit:

- [ ] Alle `std::vector` parameters zijn `const &` (tenzij je ze wijzigt)
- [ ] Geen waarschuwingen bij compileren (`-Wall -Wextra`)
- [ ] Alle variabelen geïnitialiseerd
- [ ] Loops gebruiken `size_t` of range-based for
- [ ] Geen layer/matrix copies in loops (gebruik references)
- [ ] Functie doet maar één ding (Single Responsibility)
- [ ] Zinvolle variabele namen (geen `x`, `temp`, `data2`)
- [ ] Error handling (throw exception bij invalid input)

---

## 📚 Nuttige Resources

### Backpropagation begrijpen:
- 3Blue1Brown video series on neural networks (YouTube)
- "Neural Networks and Deep Learning" by Michael Nielsen (free online book)

### C++ leren:
- cppreference.com - officiële C++ documentatie
- learncpp.com - beginner-friendly tutorials
- "Effective Modern C++" by Scott Meyers (book)

### MNIST:
- http://yann.lecun.com/exdb/mnist/ - dataset
- IDX file format beschrijving op Yann LeCun's site

---

## 🎯 Snelle Start - Volgende Sessie

1. **Fix kritieke bugs** (30 min)
   - Layer copy bug in forward pass
   - Softmax max initialisatie
   - Matrix addition loop

2. **Implementeer ReLU derivative** (15 min)
   - Voeg functie toe aan Layer class
   - Test met handmatig voorbeeld

3. **Gradient storage toevoegen** (30 min)
   - Update Layer class met nieuwe members
   - Update forward pass om z op te slaan

4. **Begin backward pass voor één layer** (1-2 uur)
   - Implementeer basis versie
   - Test met simpel voorbeeld

Good luck! 🚀
