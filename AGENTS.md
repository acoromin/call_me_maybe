# Call Me Maybe — contexto y hoja de ruta

## Objetivo y forma de trabajar

Construir el proyecto obligatorio y, después, aspirar a todos los bonus del subject.
El objetivo de aprendizaje tiene la misma importancia que el programa terminado:
el estudiante debe poder explicar cada decisión y seguir una generación token a token.

### Regla de autorización — prioridad máxima

- No escribir, modificar ni borrar código del proyecto salvo que el estudiante
  lo pida explícitamente. Esto incluye arreglos pequeños y código de prueba.
- Se puede mantener este `AGENTS.md` para conservar contexto, reglas y el plan.
  Leer código, analizarlo y proponer cambios también está permitido.
- No ejecutar comandos que escriban archivos de código en el proyecto sin una
  orden explícita.
- Frases como «¿qué harías?», «explícame» o «empezamos» no autorizan a cambiar
  código. Antes de implementar, explicar el problema, el cambio concreto y su
  motivo, y dar al estudiante la oportunidad de entenderlo.
- Si el estudiante da una orden explícita para escribir código, limitarse a su
  alcance. No aprovecharla para arreglos adicionales. Mostrar después qué cambió
  y cómo se comprobó, sin ocultar pasos ni dar por supuesto que se entendieron.

### Necesidad pedagógica actual

El estudiante ya entiende los conceptos básicos de token, ID, tokenizador, logit,
`encode`, `decode` y el SDK. La dificultad actual es **cómo convertir esos conceptos
en una implementación**. No repetir definiciones generales ni describir el `main.py`
provisional como si eso enseñara el algoritmo.

- No pedirle que cree una función o archivo para resolver un problema que aún no
  se ha construido con él. Decir «haz una función que devuelva los IDs válidos»
  no es una explicación suficiente.
- Antes de cualquier ejercicio de programación, hacer un recorrido completo y
  concreto: objetivo de una pieza, entradas y salida, datos que conserva, pasos
  internos, condiciones, finalización y un ejemplo trabajado de principio a fin.
- Después explicar la traducción a Python: qué archivo tendría sentido, qué
  funciones, parámetros, tipos y estructuras de datos se usarían, qué hace cada
  bucle o condición y por qué. Guiar línea a línea mientras el estudiante escribe;
  no entregar código terminado ni modificarlo por él salvo petición expresa.
- Hacer una sola pieza pequeña cada vez y comprobar comprensión mediante una
  conversación sobre el ejemplo, no con preguntas triviales ni tareas abiertas.
- El estudiante está frustrado por las explicaciones vagas y los cambios de
  código no solicitados. Dar instrucciones directas y concretas; reconocer los
  límites de lo que aún no se ha decidido.

- Actuar como profesor y colaborador. Empezar por un caso concreto del proyecto,
  introducir un concepto cada vez y comprobar que se entiende antes de ampliar.
- Antes de una decisión de diseño, explicar el problema, la solución sencilla, una
  alternativa relevante, el coste de cada una y su relación con el subject.
- Hacer cambios pequeños y revisables. Mostrar qué cambió, por qué, cómo se probó y
  qué queda pendiente. No entregar una máquina de estados completa de golpe.
- Los repositorios ajenos son material de estudio, no código para copiar ni pruebas
  de corrección. Verificar sus ideas con el SDK y con casos nuevos.
- Si el estudiante señala que una explicación es abstracta, volver a una entrada,
  unos IDs, unos logits y una salida observables antes de seguir.

## Fuentes y restricciones

El [subject local](en.subject.pdf), versión 1.5, es la fuente de verdad. El SDK público
es la única interfaz permitida para usar Qwen. El programa propio no debe importar
PyTorch, Transformers, Hugging Face, Outlines ni acceder a miembros privados del SDK.
Las clases propias deben usar Pydantic para validación. El conjunto de funciones y
peticiones puede cambiar durante la evaluación; no codificar respuestas de ejemplo.

Contrato obligatorio:

- Python 3.10+, `uv sync`, `uv run python -m src`, opciones
  `--functions_definition`, `--input`, `--output` y rutas por defecto.
- Archivo de salida por defecto `data/output/function_calling_results.json` con un
  array de objetos que tengan exactamente `prompt`, `name` y `parameters`.
- La función la elige Qwen mediante logits restringidos a nombres disponibles.
  Los argumentos se extraen con el modelo y respetan los tipos definidos.
- Constrained decoding durante la generación: cada token elegido debe conservar
  una ruta a un JSON y esquema válidos. Validar o reparar solo al final no basta.
- Objetivos del subject: 100 % de salidas JSON conformes, al menos 90 % de acierto
  en función y argumentos, y menos de 5 minutos para las pruebas en hardware estándar.
- Manejo claro de errores, `Makefile`, `flake8`, `mypy`, README en inglés y
  los archivos de entrega. No incluir `data/output/` en el repositorio.

Una salida puede ser estructuralmente válida y semánticamente equivocada. Medir ambas
cosas por separado. El modelo propone contenido; nuestro programa controla las
opciones válidas y la estructura. No calcular la función con palabras clave o `if`.

## Punto de partida verificado (2026-11-06)

- `data/input/` contiene 5 definiciones y 11 peticiones. Ambas listas se leen bien.
- `src/models.py` contiene modelos Pydantic iniciales. Los tipos de parámetros aún
  son cadenas libres y el modelo de salida no comprueba que los parámetros coincidan
  con la función elegida.
- `src/json_parser.py` lee y valida JSON correcto. Se corrigió la importación de
  `ValidationError`; ahora un archivo `{` produce un `ValueError` controlado en
  ambos cargadores. Las 5 funciones y 11 peticiones siguen leyéndose bien.
- `main.py` construye contexto, llama al SDK y genera 50 tokens con `argmax` libre.
  Es una prueba de inferencia; todavía no restringe tokens, no detecta el final de
  un objeto ni escribe la salida requerida. Su ejemplo fijo `fn_greet` puede no
  existir si cambian las definiciones.
- `src/token_choice.py` es un primer helper aislado: elige el mayor logit entre
  IDs permitidos. Se probó con datos pequeños; aún no está conectado al modelo.
- `python -m src` falla porque falta `src/__main__.py`. Falta `Makefile` y README.
  `pyproject.toml` declara Python `>=3.13`; revisar compatibilidad con el mínimo
  `3.10` del subject. La instalación local funciona, pero falta comprobar la
  entrega desde un entorno limpio.
- Hay cambios previos del estudiante en el árbol de trabajo. Preservarlos; no
  resetearlos ni sustituirlos al organizar el proyecto.

## Referencias estudiadas y lecciones

| Repositorio | Idea útil | Límite observado / comprobación pendiente |
| --- | --- | --- |
| [SPujol98](https://github.com/SPujol98/call-me-maybe) | Estructura fija en cola; nombres mediante prefijos de IDs; omite la llamada al modelo si solo queda un token. | `monitor.py` concentra muchas responsabilidades. El prompt se escapa reemplazando solo comillas, por lo que barras invertidas o controles requieren comprobación. No asumir que su afirmación de 100 % está demostrada. |
| [eepylaurie](https://github.com/eepylaurie/call-me-maybe) | Separa restricciones (`constraints.py`) del bucle (`decoder.py`); simula cada token completo antes de aceptarlo, útil con tokens que contienen varios caracteres. | Recorre candidatos del vocabulario por paso y su código declara una suposición sobre nombres donde uno no es prefijo de otro. Medir coste y probar nombres nuevos. |
| [MatheoGerard](https://github.com/MatheoGerard/Call-me-maybe) | Separa selección de función y generación de valores; validadores pequeños para prefijos, números y cadenas. | Estudiar si sus conversiones finales mantienen el contrato. Separar fases ayuda a aprender, pero hay que demostrar constrained decoding real en cada elección. |
| [cperales-42](https://github.com/cperales-42/call-me-maybe) | Gramática más amplia; distingue tokens forzados de decisiones del modelo y contempla eventos, tokenizador y caché. | Mucho código para empezar. Usarlo como referencia de casos límite y bonus, no como arquitectura inicial. |

Estas son observaciones del código publicado, no resultados de ejecutar sus pruebas.
Revisar de nuevo los archivos concretos antes de adoptar una técnica.

## Arquitectura preparada para los bonus

Flujo: archivos y CLI → modelos/validación → contexto → estado de restricciones
→ elección del siguiente token con logits del SDK → objeto validado → JSON de salida.

Fijar las fronteras entre piezas desde el principio, aunque cada pieza empiece con
una implementación pequeña. La primera versión obligatoria usará estas mismas
fronteras; los bonus ampliarán su comportamiento.

| Pieza y contrato inicial | Mandatory | Ampliación prevista |
| --- | --- | --- |
| Acceso al modelo/tokenizador: `encode`, `decode`, `next_logits`, archivos públicos del vocabulario | Un único adaptador alrededor de `Small_LLM_Model`; el resto del programa no conoce detalles del SDK. | Otro modelo o tokenizador propio detrás del mismo contrato; aquí medir la viabilidad de caché y batching. |
| Reglas de salida: `allowed_next`, `advance`, `is_complete` sobre un estado explícito | Reglas derivadas de las funciones recibidas; nombres, claves y tipos válidos. | Extender reglas de valores a objetos/listas anidados sin cambiar el bucle de generación. |
| Bucle de generación: contexto → reglas → logits → elección → avance | Construye la llamada paso a paso y se detiene al completar el objeto. | Trazas, visualización, límites y recuperación de errores como observadores y políticas alrededor del mismo bucle. |
| Elección entre IDs permitidos | `argmax` restringido, usando `src/token_choice.py` o su evolución. | Otras políticas solo si una medición las justifica; no alterar quién decide la función. |
| Entrada/salida y validación final | CLI, Pydantic, JSON exacto y errores claros. | Esquemas y pruebas más complejos sin mezclar lógica de archivos con logits. |

Estas fronteras son contratos de diseño, no una orden de crear cinco clases nuevas.
Si se introduce una clase propia, respetar el requisito de Pydantic. Empezar con
funciones y datos sencillos cuando basten.

Decisiones que sí tomamos ahora:

- Las restricciones salen de las definiciones de entrada, nunca de los cinco
  ejemplos conocidos ni de palabras clave del prompt.
- El generador pregunta a las reglas qué IDs son válidos; no contiene `if` para
  cada nombre de función o cada parámetro.
- El acceso a tokenización y logits queda en una frontera única. El código de
  restricciones no depende de atributos privados del SDK.
- Cada paso puede registrar un evento pequeño (estado, opciones, ID elegido y si
  se llamó al modelo). El registro será opcional para no encarecer el mandatory.
- El estado representa la posición actual en el esquema, no una lista fija de
  `param1`, `param2`, etc. Así puede ampliarse a valores anidados.
- Las definiciones se traducirán a una representación de tipos que pueda crecer;
  evitar que el bucle dependa de `str | int | float | bool` como lista cerrada.
  El formato exacto de tipos anidados se decidirá cuando haya ejemplos y pruebas.

Decisiones que necesitan experimentos antes de fijarse:

- Representar continuaciones por prefijos de IDs, por texto decodificado o de
  forma híbrida. Comprobar tokens con varios caracteres, escapes y límites entre
  fragmentos codificados.
- Forzar texto estructural tokenizado o construir parte del JSON con un serializador.
  En ambos casos, la elección de función y valores debe seguir siendo del modelo
  bajo restricciones reales y cumplir el requisito del subject.
- Caché: identificar prefijos reutilizados y medir ahorro. El SDK público actual
  recibe una lista de IDs y devuelve logits de una sola secuencia; no ofrece caché
  interna ni API de batch. No diseñar alrededor de una capacidad inexistente.

Mantener separadas, al menos conceptualmente, estas responsabilidades:

1. Entrada y contrato: archivos, tipos, errores y rutas.
2. Modelo y tokenizador: usar solo métodos públicos del SDK.
3. Restricciones: dado el texto/estado parcial, obtener continuaciones válidas.
4. Elección: Qwen puntúa; el programa elige el mayor logit válido.
5. Resultado y medición: validar, escribir y medir validez, acierto y tiempo.

Comparar las representaciones de restricciones con tokens reales del SDK, nombres
compartidos, comillas, números y cadenas escapadas. Un
`argmax` sobre IDs permitidos equivale, para esa elección, a poner `-inf` en los
demás; cuando hay pocos IDs evita copiar todo el vector. La mayor ganancia probable
será evitar llamadas al modelo cuando solo haya una continuación, pero medir antes.

## Etapas, en orden

### 0. Consolidar la base

Corregir el `NameError` y añadir el punto de entrada `python -m src` sin rediseñar
todavía el programa. Confirmar que se leen las 5 funciones y 11 peticiones, y que
un archivo roto da un error claro. Explicar la diferencia entre leer JSON válido y
producir llamadas válidas. Dejar CLI, rutas y validación final para la etapa 4.

**Avance:** las entradas correctas se leen, las incorrectas dan un error claro y
el comando obligatorio arranca.

### 1. Ver una decisión real del modelo

Con dos funciones y una petición, observar `encode`, IDs, logits y `decode`.
Construir/usar una lista pequeña de IDs permitidos y comprobar que el elegido viene
de Qwen. Empezar por el nombre de la función, sin generar aún todo el JSON.

**Avance:** el estudiante puede seguir una traza: contexto → logits → opciones
válidas → ID elegido → texto; no hay heurística sobre el prompt.

### 2. Restringir nombres con definiciones variables

Generar un nombre completo que pertenezca a las definiciones recibidas. Probar
nombres con prefijo común, nombres nuevos y distintas tokenizaciones. Resolver
explícitamente cómo se detecta el final del nombre y qué pasa si no hay opción válida.

**Avance:** ningún nombre inventado y la elección cambia según logits y petición.

### 3. Restringir valores por tipo

Después de elegir la función, usar sus claves y tipos. Abordar primero un número,
luego cadena y booleano; añadir cero o varios parámetros. Las claves conocidas se
escriben desde la definición. Probar números negativos/decimales, escapes, cadena
vacía y límites de longitud. Evitar cierres prematuros o bucles sin fin.

**Avance:** valores parseables y del tipo exigido para definiciones no vistas.

### 4. Completar el JSON y el programa obligatorio

Construir el objeto exacto con `prompt` original escapado como JSON, `name` elegido
y todos los `parameters`. Finalizar cuando el objeto esté completo. Validar el
resultado, escribir el array y manejar fallos sin perder silenciosamente una petición.
Integrar CLI, Makefile y empaquetado.

**Avance:** salida 100 % parseable y conforme al esquema en una batería diversa.

### 5. Medir y mejorar

Separar validez, acierto de función, acierto de argumentos, errores y tiempo por
petición. Medir en el Slimbook con ejemplos y definiciones nuevas. Ajustar contexto
y restricciones solo después de observar fallos. Comparar coste de recorrer vocabulario,
usar prefijos/cache y omitir llamadas forzadas.

**Avance:** evidencia para el objetivo de 90 % y 5 minutos; limitaciones documentadas.

### 6. Bonus funcionales, uno a uno

Implementar y comprobar, sin romper el obligatorio:

1. Suite amplia de pruebas y visualización/traza de cada elección.
2. Recuperación avanzada de errores.
3. Caché y, si la interfaz pública lo permite, batching. No prometer batching sin
   verificarlo: el SDK actual expone logits para una sola lista de IDs.
4. Argumentos anidados.
5. Varios modelos, manteniendo Qwen/Qwen3-0.6B operativo.
6. Tokenizador propio: recodificar `encode` y opcionalmente `decode` a partir de los
   archivos públicos; demostrar su integración con las restricciones.

El orden puede cambiar según dependencias y mediciones. Cada bonus se añadirá en
la pieza prevista en la tabla de arquitectura. Cada bonus declarado como hecho debe
funcionar y poder demostrarse; no basta con mencionarlo en README.

### 7. Documentación y defensa

README en inglés con algoritmo, decisiones, pruebas, rendimiento, dificultades,
uso y papel de la IA. Ensayar una traza completa y una pequeña modificación.

## Cómo abordar cada sesión

Elegir una sola etapa y un caso mínimo. Primero observar; después explicar el
comportamiento; luego comparar dos soluciones y elegir una; finalmente hacer un
cambio pequeño y comprobarlo. Registrar aquí los cambios de etapa o decisiones
importantes, no cada conversación.

**Siguiente paso:** retomar con una lección guiada de implementación, sin tocar
código: construir de principio a fin, con datos concretos, la lógica que obtiene
las continuaciones válidas de un nombre de función; explicar después cómo se
traduciría a Python, paso a paso. Solo cuando el estudiante entienda y pida
escribir código, implementarla. El `NameError` del parser ya se corrigió;
`src/token_choice.py` fue creado prematuramente por el asistente, no está conectado
al programa y el estudiante puede decidir borrarlo. La entrada `python -m src`
sigue pendiente y puede abordarse más adelante.
