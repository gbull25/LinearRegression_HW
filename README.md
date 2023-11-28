# LinearRegression_HW
HW project (MOVS 2023, ML course). With all sorts of linear models and FastApi service.

## Сервис развернут на render. 
## [Сcылка на сервис](https://linear-regression-cars.onrender.com/ "Сcылка на сервис") 
    - Данные для тестовых прогонов лежат в папке test_csv. 
    Это исходный тестовый датасет из задания и его кусочки. 
    Сами данные в датасете не были изменены.

## Результаты работы

### Предобработка данных и EDA

- В первую очередь были очищены столбцы mileage, engine, max_power (от единиц измерения). Это сделано для того, чтобы их можно было преобразовать к числовому типу. Для этого пишу функции с использованием регулярных выражений. 

- С признаком torque вышла забавная ситуация, сначала я его удалил, но после того, как получил неудовлетворительные метрики качества построенных моделей, решил с ним все-таки разобраться. Torque очень неприянтый признак, у него есть два значения с разными единицами
 измерения, которые еще и отличаются от строки к строке. Чтобы с ним разобраться, пишем отдельную функцию, которая преобразуют колонку в torque и max_torque_rpm с конвертированными и округленными данными для различных единиц измерения. 

- Удаляем дубликаты строк, предварительно откладывая в сторону таргет (стоимость автомобиля), чтобы удалить только те строки, где совпадают все значения кроме цены.

- Пропуски заполняем медианными значениями.

- Визуализируем данные, получается такой небольшой EDA. Вот некоторые из наблюдений из этой части домашки:
    - Судя по графикам попарных распределений cамая сильная связь с таргетом у признака max power. Признаки year и engine имеют линейную зависимость с целевой переменной. Зависиость между признаками year и целевой переменной выглядит квадратичной
    - Признаки engine, torque и seats сильно связаны, есть связь между признаком year и признаками mileage, km_driven, max power. Связаны также engine и torque (это довольно очевидно :)).

### Модель только на вещественных признаках

- Обучаем классическую линейную регрессию, стандартизацуем фичи. Получаем результат:
    - Значение MSE для теста:  232795188932.2784
    - Значение R^2 для теста:  0.595018050775644
    - Среднее кросс-валидации: 0.56901
    - Наиболее информативным оказался признак max_power.
    - Здесь проведена стандартизация признаков. Далее все модели гоняются на них.

- Обучеаем Lasso-регрессию:
    - Значение MSE для стандартизированного теста:  232795880763.04
    - Значение R^2 для стандартизированного теста:  0.5950168472328574
    - Среднее кросс-валидации: 0.56902
    - Веса не занулились

- Перебором по сетке (c 10-ю фолдами) подбираем оптимальные параметры для Lasso-регрессии:
    - Получаем: alpha = 19900
    - Значение MSE для стандартизированного теста:  246090597713.12683
    - Значение R^2 для стандартизированного теста:  0.5718887043810801
    - Среднее кросс-валидации: 0.57573
- Перебором по сетке (c 10-ю фолдами) подбираем оптимальные параметры для ElasticNet-регрессии:
    - Значение MSE для стандартизированного теста:  248544444699.02045
    - Значение R^2 для стандартизированного теста:  0.5676198715928964
    - Среднее кросс-валидации: 0.57605
    - Получаем: alpha = 0.96, l1_ratio=0.86 


### Добавляем категориальные фичи

- Кодируем категориальные фичи (и seats) методом OneHot-кодирования
- Перебираем параметр регуляризации alpha для гребневой (ridge) регрессии:
    - Получаем:
    - alpha = 3
    - Значение MSE для стандартизированного тесте:  214860632342.3589
    - Значение R^2 для стандартизированного тесте:  0.6262178866467067
    - Среднее кросс-валидации: 0.61708
    - Видим, что качество предсказаний улучшилось довольно заметно!

### Feature Engineering

- Добавляем полиминальные признаки второй степени

    - Значение MSE для теста:  101582475540.26753
    - Значение R^2 для теста:  0.823282134222711
    - Среднее кросс-валидации: 0.61708
- Пробуем работать с имеющимися признаками

    - Применяем Indicator Method. Добавляем столбец с индикатором пропусков в данных. Но для начала пройдемся по этапам предобработки данных заново, чтобы ничего не напутать.
    - После добавления индикаторов пропусков в данных пробуем эти самые пропуски заполнять нулями и медианами. Пытаемся уловить разницу.
    - Разница не обнаружена :(

        - При заполнении нулями:
            - Значение MSE для теста:  215700949843.863
            - Значение R^2 для теста:  0.62475602903149
            - Среднее кросс-валидации: 0.57693

        - При заполнении медианами:
            - Значение MSE для теста:  215700949843.85007
            - Значение R^2 для теста:  0.6247560290315124
            - Среднее кросс-валидации: 0.57691
    - Попробуем добавить One-hot кодирование к индикации пропусков:
        - Значение MSE для теста:  199746714082.18774
        - Значение R^2 для теста:  0.6525108014852616
    - Логаримфмируем целевую переменную
    - Добавим некоторые полиномильные признаки - логарифмируем признаки year, engine, km, torque (оба). Пробуем делить max_power на engine и engine на torque.
    - Смотрим что получилось:
        - Значение MSE для теста:  0.09308288320360683
        - Значение R^2 для теста:  0.8698700523438925
        - Среднее кросс-валидации: 0.83625
        - Круто! 
    - Попробуем все сложить, One-hot + Индикация пропусков + Полиноминальные признаки + Логаримфирование:
        - Значение MSE для теста:  0.08494149361736263
        - Значение R^2 для теста:  0.8812517217147107
        - Среднее кросс-валидации: 0.84147
        - Небольшой прирост, слишком много признаков
    - Откатимся немного назад и попробуем использовать марку машины. Их не так много, поэтому можно попробовать закодировать. Берем первое слово из признака name
    - В трейне не хватает производетелей Опель и Ашок. Брутфорсим :) 
    - Т.к. бренд Maruti представлен больше других в обоих частях датасета, изменим две строчки на нужные нам марки. Они не должны сильно повлиять на качество модели
        - Значение MSE для теста:  0.058073674602835176
        - Значение R^2 для теста:  0.9188129549045604
        - Среднее кросс-валидации: 0.88432
    - Пробуем дополнительно добавить полиноминальные признаки:
        - Значение MSE для теста:  0.05543276831864425
        - Значение R^2 для теста:  0.922504944072695
        - Среднее кросс-валидации: 0.88740
        - На этом закончим. Результат мне показался достаточного уровня.

### Решаем бизнес-задачу
- Не забываем убрать логарифм с предсказания и таргета
- Доля предиктов, отличающихся не более чем на 10% от реальной цены в тесте - 38%.

### Проделываем все от начала и до самого конца (получения итоговой модели)
- Страдаем, листая километровый ноутбук. Находим косяки, устраняем, собираем все в цельный пайплайн.

### Реализация сервиса FastAPI
- Проделываем слеудющее:
    - Средствами pydantic описываем класс базового объекта (заготовка)
    - Класс с коллецией объектов (заготовка)
    - Метод post, который получает на вход один объект описанного класса в формате json
    - Метод post, который получает на вход коллекцию объектов описанного класса в формате csv

- В процессе реализации сервиса я столкнулся с двумя вопросами по начинке поступающих на вход данных:
    - Что делать с пропусками (должны ли они быть)?
    - Что делать с признаком torque (должны быть две колонки на входе или делить на две уже внутри сервиса)?

- Мне хотелось, чтобы сервис умел работать с исходными данными, в том числе мог пережевывать тестовый датасет целиком.
    - Поэтому я решил заполнять пропуски в данных с помощью медиан, полученных на трейне в ноутбуке и экспортированных в .pickle файл.
    Не самое элегантное решение, но мне другого в голову не пришло. 
    - Со столбцом torque, соответственно такая ситуация: я ожидаю, что на 
    вход сервису будет поступать информация в изначальном виде, одной колонкой.

- В итоге на вход методу post, работающему с форматом csv, можно подавать тестовый датасет целиком (придется подождать) или частично
без какой-либо предобработки. 

### Картинки

![Изображение](https://github.com/gbull25/LinearRegression_HW/raw/main/pics/handle_1.png "Первая ручка")

![Изображение](https://github.com/gbull25/LinearRegression_HW/raw/main/pics/handle_2.png "Вторая ручка")