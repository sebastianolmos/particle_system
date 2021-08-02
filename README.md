# Particle System using CUDA

## Descripción
Proyecto final del curso CC7515-1 Computación en GPU. Corresponde a una escena que simula colisiones de esferas en un espacio de 3 dimensiones, encerradas en una caja.
Para el calculo de colisiones se utilizo la herramienta CUDA con el lenguaje de programacion C++, basandose en en el articulo [Particle Simulation using CUDA](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf) de Simon Green. Ademas se utiliza de referencia la simulacion que se encuentra en los ejemplos de CUDA Toolkit. 

## Librerías usadas
Se uso el lenguaje C++ con la siguientes librerias:
- [Glad](https://glad.dav1d.de/) : Libreria necesaria para cargar los punteros a funciones de OpenGL. En este proyecto se uso OpenGL 4.5
- [GLFW3](https://www.glfw.org/) : Libreria usada con OpenGl que provee una API para manejar ventanas
- [GLM](https://glm.g-truc.net/0.9.9/index.html) : Libreria con funciones matematicas utiles para el uso de aplicaciones con OpenGL
- [Dear Imgui](https://github.com/ocornut/imgui): Libreria para poder agregar un menu configurable

## Como se instalaron las librerías
A continuación se darán los pasos con las que se pudo instalar las diferentes librerías para poder usarlas en el programa Visual Studio 2019:
- Se tiene que configurar el proyecto en VS, creando las carpetas /Libraries/include y /Libraries/lib si no se encuentran
- Seleccionar la plataforma x64 en el editor
- Ir a configuraciones del proyecto y seleccionar en Platform: All Platforms
- Ir a VC++ Directories -> Include Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/include -> ok
- Ir a VC++ Directories -> Library Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/lib -> ok
- Ir a Linker -> Input -> Additional Dependencies -> Edit -> poner en el campo de texto:
```
glfw3.lib
opengl32.lib
```
Luego para instalar las diferentes librerías:
- [Glad](https://glad.dav1d.de/) : Descargar la version OpenGL/GLAD (version 4.5 Core), abrir glad.zip -> ir a /include y copiar carpetas "glad" y "KHR" a la carpeta del proyecto /Libraries/include. Del mismo zip -> ir a /src y copiar el archivo "glad.c" en la carpeta raíz del proyecto.
- [GLFW3](https://www.glfw.org/) : Descargar, y compilar con Cmake en una carpeta build, ir a ../build/src/Debug y copiar el archivo "glfw3.lib" a la carpeta del proyecto Libraries/lib. Ir a ../include y copiar la carpeta "GLFW" a la carpeta del proyecto Libraries/include
- [GLM](https://glm.g-truc.net/0.9.9/index.html) : Descargar, descomprimir y copiar directorio que sea raíz de glm.h y pegarla en Libraries/include
- [Dear Imgui](https://github.com/ocornut/imgui): Descargar los archivos y descomprimir. incorporar los archivos del directorio base y de la versión a usar (openGL3.* y glfw, headers y cpp que se encuentran en la carpeta backends) al proyecto directamente. Importante utilizar la versión que soporta Docking

## Como se instalaron las librerías
Para poder ejecutar el proyecto debe tener una GPU compatible con CUDA. La aplicacion se puede ejecutar desde el editor Visual Studio 2019, seleccionando la opcion Realease con la plataforma x64 o.

## Controles:

Los controles de teclado son:
- [SCAPE] Salir de la aplicación
- [SPACEBAR] Posiciona la camara al centro

Los Controles con el mouse son:
- [Click izquierdo + Movimiento del mouse] Permite rotar la escena
- [Click derecho + Movimiento del mouse] Permite desplazar la escena o desplazar una esfera, dependiendo de la Checkbox "Object to Collide?" del menu 
- [Scroll] Permite alejar o acercar la escena

Aparte con el mouse se puede controlar el menú que aparece en la bentana

## Como usar el menú de Imgui:
Basta con posicionar el mouse sobre el menú o las pestañas y seleccionar las diversas opciones con el click izquierdo.
