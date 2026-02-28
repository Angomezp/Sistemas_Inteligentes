# Sistemas Inteligentes

**Integrantes / Grupo**

- Grupo: Azucar Morena
- Integrantes: 
    - [Leidy Yulliana Quiñones Gonzalez](https://github.com/LeidyQG)
    - [Angel David Gomez Pastrana](https://github.com/Angomezp)

---

**Descripción**

Repositorio para proyectos y ejercicios relacionados con la materia Sistemas Inteligentes. Este README  contiene una descripción mínima del contenido actual y marcadores para documentar nuevas carpetas o módulos que se añadan en el futuro.

**Estructura**

```
.
├── README.md                
├── .gitignore
└── Canibales_Misioneros/ #Bot del juego Canibales y Misioneros
    ├── bot_AzucarMorena.py  # Script principal 
    ├── area_config.json      # Archivo de configuración generado por calibración (local)
    └── sprites/             # Imagenes
```

- `Canibales_Misioneros/` : Bot para el puzzle "Misioneros y Caníbales" — captura la región del juego, detecta misioneros, caníbales y la balsa mediante plantillas y automatiza clicks para intentar resolver el puzzle.

Nota: el archivo `Canibales_Misioneros/area_config.json` se genera localmente cuando se calibra el área de juego. Por seguridad y por configuración del repositorio puede estar marcado como no versionado (no incluido en git). Si clonas el repositorio por primera vez, ejecuta la opción de calibrado en el script para generar este archivo antes de usar el bot.

