## Prepare your environment
*แนะนำให้สร้าง virtual environment เพื่อไม่ให้กระทบต่อ environment ของเพื่อนๆนะครับบ*
1. install virtualenv
```bash
pip install virtualenv
```
2. create virtual environment
```bash
virtualenv <env name>
```
3. activate virtual environment
- windows
    ```bash
    <env name>\Scripts\activate
    ```
- macOS or linux
    ```zsh
    source <env name>/bin/activate
    ```
5. install package from requirements
```bash
pip install -r open3d_requirements.txt
pip install -r requirements.txt
```