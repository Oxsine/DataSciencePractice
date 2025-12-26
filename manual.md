
# Активация виртуальной среды и её создание

---

## Первоначальная настройка среды и её активация (через скрипт)

1. Активируйте `startup.ps1`, если вы на Windows, для создание виртуальной среды, для дистрибутивов Linux и MacOS активируйте `startup.sh`.
2. После создание виртуальной среды под именем `.venv_ds` в консоли будет отображаться такая приписка `(.venv_ds)`

---

## Первоначальная настройка среды и её активация (самостоятельно)
1. Для создания виртуальной среды, вы прописываете следующию команду:

   * __Windows__:
    ```powershell
    python -m venv .venv_ds
    ``` 
   * __Дистрибутивы Linux/MacOS__:
    ```bash
    python3 -m venv .venv_ds
    ``` 
2. Активации `.venv_ds`:

    * __Windows__:
    ```powershell
    ./.venv_ds/Scripts/Activate.ps1
    ``` 
   * __Дистрибутивы Linux/MacOS__:
    ```bash
    source ./.venv_ds/bin/activate
    ``` 
3. Загрузка библиотек через `requirements.txt`

   * __Windows__:
    ```powershell
    pip install -r requirements.txt
    ``` 
   * __Дистрибутивы Linux/MacOS__:
    ```bash
    pip3 install -r requirements.txt
    ``` 

## Советы
- Для выхода из виртуальной среды используйте команду `deactivate`
- Для создание файла `requirements.txt` используйте команду:
   * __Windows__:
    ```powershell
    pip freeze > requirements.txt
    ``` 
   * __Дистрибутивы Linux/MacOS__:
    ```bash
    pip3 freeze > requirements.txt
    ``` 
- Если вы находитесь на Windows и у вас не получаеться активировать среду, то пропишите эту команду ```Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process```
- Обновление `pip`:
   * __Windows__:
    ```powershell
    python -m pip install --upgrade pip
    ``` 
   * __Дистрибутивы Linux/MacOS__:
    ```bash
    python3 -m pip install --upgrade pip
    ```  
- Вывод списка библиотек в среде:
  * __Windows__:
    ```powershell
    pip list
    ``` 
  * __Дистрибутивы Linux/MacOS__:
    ```bash
    pip3 list
    ``` 
