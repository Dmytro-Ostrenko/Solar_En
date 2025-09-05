# Solar Energy Repository

## English Version

### Overview
This repository, "Solar Energy," contains the results and materials from two scientific articles focused on the application of artificial neural networks (ANN) in solar energy systems, specifically for photovoltaic stations (PVS). The materials include PDF versions of the articles and Jupyter notebooks (.ipynb) demonstrating the neural network models used in the research. These tools aim to improve the efficiency, reliability, and optimization of solar power generation through computer vision and machine learning techniques.

The repository is designed for developers, researchers, and enthusiasts in renewable energy who want to explore or replicate the models. No prior knowledge of the details is required—the descriptions below provide a clear guide.

### Articles and Key Findings
1. **Article 1: Analysis of Convolutional Neural Networks for Improving Reliability in PVS Operation**  
   - **Filename**: Article1.pdf  
   - **Authors**: O. Yu. Kollarov, D. O. Ostrenko  
   - **Publication**: Scientific Works of DonNTU, Series: "Electrical Engineering and Energy," No. 2(31)'2024  
   - **Summary**: This article explores the use of Convolutional Neural Networks (CNN) for automatic recognition of photovoltaic panel states based on images. It analyzes the impact of hyperparameters (e.g., number of layers, kernel size, learning rate, activation functions) on classification accuracy for defects like physical damage, electrical faults, and dust accumulation. The study compares CNN with traditional ML models (e.g., SVM, Random Forest) and demonstrates how optimized CNN can enhance PVS maintenance and energy efficiency. Key results include high accuracy (up to 96.6% with transfer learning) and recommendations for hybrid models like CNN-LSTM for forecasting generation under variable weather.

2. **Article 2: Application of ANN for Insolation Assessment and PVS Power Optimization**  
   - **Filename**: Article2.pdf  
   - **Authors**: O. Yu. Kollarov, D. O. Ostrenko  
   - **Publication**: Scientific Works of DonNTU, Series: "Electrical Engineering and Energy," No. 1(32)'2025  
   - **Summary**: This work investigates ANN for analyzing sky images to classify weather conditions, locate the sun, and estimate illumination levels without physical sensors. A simple fully connected neural network is built for sky state classification (e.g., clear, cloudy, dark) using features like cloud cover and brightness. Additional algorithms detect sun coordinates and relative insolation. The research highlights applications for dynamic panel tilt adjustment and short-term generation forecasting, comparing ANN with models like Random Forest and XGBoost. Results show over 95% accuracy in classification and stable performance without overfitting.

### Jupyter Notebooks
These notebooks implement the models described in the articles. They require Python 3.x, libraries like TensorFlow/Keras, OpenCV, NumPy, Pandas, and Matplotlib. Install dependencies via `pip install -r requirements.txt` (if provided) or manually.

1. **Solar Panel Status Identification**  
   - **Filename**: Aricle1_Solar panel status identification.ipynb  
   - **Description**: This notebook builds and trains a CNN model for classifying photovoltaic panel states (e.g., clean, damaged, dusty, electrical defects). It includes data preprocessing, model architecture, training with hyperparameters tuning, and visualization of accuracy/loss graphs. Use it to replicate defect detection experiments from Article 1.

2. **Sun Localization and Insolation Estimation**  
   - **Filename**: Article2_sun_localization.ipynb  
   - **Description**: This notebook demonstrates image processing and ANN for sky analysis: classifying weather, locating the sun via contour detection, and calculating illumination percentage. It includes filters for false positives (e.g., reflections) and visualizes results on sample images. Ideal for reproducing insolation assessment from Article 2.

### How to Use
1. **Clone the Repository**:  
   ```
   git clone https://github.com/yourusername/Solar-Energy.git
   ```
2. **Install Dependencies**:  
   Run the notebooks in Jupyter (e.g., via `jupyter notebook`). Ensure libraries are installed:  
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
   ```
3. **Run Notebooks**:  
   - Open in Jupyter Lab or Notebook.  
   - Datasets are referenced in the code (e.g., image folders); you may need to provide your own or use placeholders.  
   - For training, adjust paths and run cells sequentially.

### License
This repository is for educational and research purposes. Articles are copyrighted by the authors and DonNTU. Code in notebooks is open for non-commercial use (MIT License).

If you have questions, contact the authors at kollarov@gmail.com or dmytro.ostrenko@gmail.com.

---

## Українська Версія

### Огляд
Цей репозиторій "Сонячна енергетика" містить результати та матеріали з двох наукових статей, присвячених застосуванню штучних нейронних мереж (ШНМ) у системах сонячної енергії, зокрема для фотоелектричних станцій (ФЕС). Матеріали включають PDF-версії статей та Jupyter-ноутбуки (.ipynb) з демонстрацією моделей нейронних мереж, використаних у дослідженнях. Ці інструменти спрямовані на підвищення ефективності, надійності та оптимізації генерації сонячної енергії за допомогою комп'ютерного зору та машинного навчання.

Репозиторій призначений для розробників, дослідників та ентузіастів відновлюваної енергетики, які бажають вивчити або відтворити моделі. Не потрібно мати попередніх знань про деталі — описи нижче надають чіткий посібник.

### Статті та Ключові Результати
1. **Стаття 1: Аналіз застосування згорткових нейронних мереж для підвищення надійності у роботі ФЕС**  
   - **Файл**: Article1.pdf  
   - **Автори**: О. Ю. Колларов, Д. О. Остренко  
   - **Публікація**: Наукові праці ДонНТУ, Серія: «Електротехніка і енергетика», №2(31)’2024  
   - **Короткий опис**: Стаття досліджує використання згорткових нейронних мереж (CNN) для автоматичного розпізнавання станів фотоелектричних панелей на основі зображень. Аналізується вплив гіперпараметрів (наприклад, кількість шарів, розмір ядра, швидкість навчання, функції активації) на точність класифікації дефектів, таких як фізичні пошкодження, електричні дефекти та забруднення пилом. Дослідження порівнює CNN з традиційними моделями МН (наприклад, SVM, Random Forest) та демонструє, як оптимізована CNN підвищує ефективність технічного обслуговування ФЕС. Ключові результати: висока точність (до 96,6% з переносним навчанням) та рекомендації щодо гібридних моделей типу CNN-LSTM для прогнозування генерації за змінних погодних умов.

2. **Стаття 2: Застосування ШНМ для оцінки інсоляції та оптимізації потужності ФЕС**  
   - **Файл**: Article2.pdf  
   - **Автори**: О. Ю. Колларов, Д. О. Остренко  
   - **Публікація**: Наукові праці ДонНТУ, Серія: «Електротехніка і енергетика», №1(32)’2025  
   - **Короткий опис**: Робота присвячена дослідженню ШНМ для аналізу зображень неба з метою класифікації погодних умов, локалізації Сонця та оцінки рівня освітленості без фізичних сенсорів. Побудовано просту повнозв’язну нейронну мережу для класифікації станів неба (наприклад, ясно, хмарно, темно) на основі ознак, таких як покриття хмарами та яскравість. Додаткові алгоритми визначають координати Сонця та відносну інсоляцію. Дослідження підкреслює застосування для динамічного регулювання кута нахилу панелей та короткострокового прогнозування генерації, порівнюючи ШНМ з моделями типу Random Forest та XGBoost. Результати: понад 95% точності в класифікації та стабільна продуктивність без перенавчання.

### Jupyter Ноутбуки
Ці ноутбуки реалізують моделі, описані в статтях. Вони вимагають Python 3.x, бібліотек на кшталт TensorFlow/Keras, OpenCV, NumPy, Pandas та Matplotlib. Встановіть залежності через `pip install -r requirements.txt` (якщо надано) або вручну.

1. **Ідентифікація стану сонячних панелей**  
   - **Файл**: Aricle1_Solar panel status identification.ipynb  
   - **Опис**: Ноутбук будує та навчає модель CNN для класифікації станів фотоелектричних панелей (наприклад, чисті, пошкоджені, запилені, електричні дефекти). Включає попередню обробку даних, архітектуру моделі, навчання з налаштуванням гіперпараметрів та візуалізацію графіків точності/втрат. Використовуйте для відтворення експериментів з виявлення дефектів зі Статті 1.

2. **Локалізація Сонця та оцінка інсоляції**  
   - **Файл**: Article2_sun_localization.ipynb  
   - **Опис**: Ноутбук демонструє обробку зображень та ШНМ для аналізу неба: класифікацію погоди, локалізацію Сонця через виявлення контурів та розрахунок відсотка освітленості. Включає фільтри для помилкових спрацьовувань (наприклад, відблиски) та візуалізацію результатів на зразкових зображеннях. Ідеально для відтворення оцінки інсоляції зі Статті 2.

### Як Використовувати
1. **Клонування Репозиторію**:  
   ```
   git clone https://github.com/yourusername/Solar-Energy.git
   ```
2. **Встановлення Залежностей**:  
   Запустіть ноутбуки в Jupyter (наприклад, через `jupyter notebook`). Переконайтеся, що бібліотеки встановлено:  
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
   ```
3. **Запуск Ноутбуків**:  
   - Відкрийте в Jupyter Lab або Notebook.  
   - Датасети вказані в коді (наприклад, папки з зображеннями); можливо, потрібно надати власні або використовувати заглушки.  
   - Для навчання налаштуйте шляхи та виконуйте клітинки послідовно.

### Ліцензія
Цей репозиторій для освітніх та дослідницьких цілей. Статті захищені авторським правом авторів та ДонНТУ. Код у ноутбуках відкритий для некомерційного використання (MIT License).

Якщо є питання, звертайтеся до авторів: kollarov@gmail.com або dmytro.ostrenko@gmail.com.
