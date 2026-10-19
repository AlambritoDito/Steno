# Probar Steno en Windows

Esta guía acompaña la edición **portable de prueba** de Steno para Windows 11 de 64 bits. La beta todavía necesita pruebas en equipos distintos antes de considerarse estable.

## Abrir la aplicación

1. Descarga el ZIP portable que te comparta el proyecto. En Windows, haz clic derecho sobre el ZIP y elige **Extraer todo**.
2. Coloca la carpeta extraída en un lugar donde puedas guardar archivos, por ejemplo **Documentos → Steno**. Conserva juntos `Steno.exe`, `resources` y los demás archivos. No abras el ejecutable directamente desde el ZIP.
3. Abre **Steno.exe**. No necesitas instalar Python ni herramientas de programación.
4. Si Windows muestra un aviso de editor desconocido, comprueba que descargaste la beta que te compartió el proyecto. Los paquetes todavía no tienen certificado público. En equipos administrados, la política de Windows puede impedir su apertura; no desactives la protección del sistema.
5. Elige español, un micrófono y el modelo **tiny** para la primera prueba. Esta descarga necesita internet. Después puedes elegir un modelo más grande.

Steno utilizará una GPU NVIDIA compatible cuando pueda inicializarla. Si no puede, utilizará el procesador y mostrará el motivo. La aplicación incluye las bibliotecas necesarias para CUDA; no instala ni cambia el driver NVIDIA. La alternativa CPU puede tardar más.

## Dónde se guarda todo

Al abrir la edición portable aparece una carpeta **Steno Data** junto a `Steno.exe`. Contiene las sesiones, modelos, preferencias y una carpeta **logs**. Los archivos de diagnóstico también pueden exportarse desde Configuración.

No borres ni muevas `Steno Data` mientras Steno está abierto. Para trasladar o respaldar esta prueba, cierra la aplicación y copia la carpeta completa. Para reemplazar los archivos de la aplicación por otra beta, conserva `Steno Data`.

El instalador normal `.exe` también está previsto: guarda los datos en la carpeta de aplicaciones del usuario de Windows. La edición portable permite tener todo junto para estas primeras pruebas.

## Una prueba breve

1. Crea una sesión con un nombre de prueba.
2. Graba una frase durante unos segundos y pulsa **Detener**. Espera a que termine **Procesando**.
3. Escribe una nota. Vuelve a grabar: debe aparecer otra toma, sin reemplazar el audio anterior.
4. Cierra Steno y vuelve a abrirlo. Comprueba que sigan la nota, la transcripción y las dos tomas.
5. Exporta Markdown o ZIP y abre el documento exportado.

También puedes transcribir un archivo de audio. Para la primera prueba usa un archivo corto que no contenga información privada.

## Si algo falla

Abre **Configuración**:

- **Abrir carpeta de logs** muestra los registros locales.
- **Exportar diagnóstico** descarga un ZIP listo para compartir con quien te proporcionó la beta. Incluye versiones, hardware, estado del motor y los últimos registros disponibles; excluye grabaciones, notas, transcripciones, modelos y credenciales.

Junto al ZIP, indica qué botón pulsaste, qué esperabas que ocurriera y aproximadamente a qué hora pasó. Si apareció un mensaje, copia su texto. No necesitas ejecutar comandos ni instalar herramientas de diagnóstico.

Los registros tienen un tamaño limitado y rotan automáticamente. La aplicación no envía el diagnóstico a ningún servidor: tú decides cuándo compartir el ZIP.
