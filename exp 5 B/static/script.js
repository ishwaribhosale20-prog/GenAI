// ============================================================
// MNIST VAE LATENT SPACE EXPLORER
// STEP 6
// ============================================================

const canvas = document.getElementById("latentCanvas");
const ctx = canvas.getContext("2d");


// ============================================================
// VARIABLES
// ============================================================

let latentPoints = [];

let selectedZ1 = 0;
let selectedZ2 = 0;

let selectedX = canvas.width / 2;
let selectedY = canvas.height / 2;

let minZ1 = -3;
let maxZ1 = 3;

let minZ2 = -3;
let maxZ2 = 3;


// ============================================================
// LOAD 70,000 LATENT POINTS
// ============================================================

async function loadLatentPoints() {

    console.log("Loading latent points...");

    try {

        const response =
            await fetch("/latent-points");

        if (!response.ok) {
            throw new Error(
                "Could not load latent points"
            );
        }

        latentPoints =
            await response.json();

        console.log(
            "Loaded points:",
            latentPoints.length
        );


        // Find ranges

        minZ1 = Math.min(
            ...latentPoints.map(
                point => point.z1
            )
        );

        maxZ1 = Math.max(
            ...latentPoints.map(
                point => point.z1
            )
        );

        minZ2 = Math.min(
            ...latentPoints.map(
                point => point.z2
            )
        );

        maxZ2 = Math.max(
            ...latentPoints.map(
                point => point.z2
            )
        );


        drawGraph();

    } catch (error) {

        console.error(
            "Error loading latent points:",
            error
        );

    }
}


// ============================================================
// LATENT → CANVAS
// ============================================================

function z1ToX(z1) {

    return (
        (z1 - minZ1) /
        (maxZ1 - minZ1)
    ) * canvas.width;

}


function z2ToY(z2) {

    return canvas.height -
        (
            (z2 - minZ2) /
            (maxZ2 - minZ2)
        ) * canvas.height;

}


// ============================================================
// CANVAS → LATENT
// ============================================================

function xToZ1(x) {

    return minZ1 +
        (
            x / canvas.width
        ) *
        (maxZ1 - minZ1);

}


function yToZ2(y) {

    return minZ2 +
        (
            (canvas.height - y) /
            canvas.height
        ) *
        (maxZ2 - minZ2);

}


// ============================================================
// DRAW GRAPH
// ============================================================

function drawGraph() {

    const width = canvas.width;
    const height = canvas.height;


    // Clear

    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    // Background

    ctx.fillStyle = "white";

    ctx.fillRect(
        0,
        0,
        width,
        height
    );


    // ========================================================
    // GRID
    // ========================================================

    ctx.strokeStyle = "#dddddd";

    ctx.lineWidth = 1;

    const gridLines = 10;


    for (
        let i = 0;
        i <= gridLines;
        i++
    ) {

        const x =
            (i / gridLines) *
            width;

        ctx.beginPath();

        ctx.moveTo(x, 0);

        ctx.lineTo(x, height);

        ctx.stroke();

    }


    for (
        let i = 0;
        i <= gridLines;
        i++
    ) {

        const y =
            (i / gridLines) *
            height;

        ctx.beginPath();

        ctx.moveTo(0, y);

        ctx.lineTo(width, y);

        ctx.stroke();

    }


    // ========================================================
    // 70,000 POINTS
    // ========================================================

    if (latentPoints.length > 0) {

        ctx.fillStyle =
            "rgba(50, 100, 200, 0.28)";


        for (
            let i = 0;
            i < latentPoints.length;
            i++
        ) {

            const point =
                latentPoints[i];


            const x =
                z1ToX(point.z1);

            const y =
                z2ToY(point.z2);


            ctx.fillRect(
                x,
                y,
                2,
                2
            );

        }

    }


    // ========================================================
    // AXES
    // ========================================================

    ctx.strokeStyle = "#222";

    ctx.lineWidth = 2;


    // Z2 = 0

    if (
        minZ2 <= 0 &&
        maxZ2 >= 0
    ) {

        const zeroY =
            z2ToY(0);

        ctx.beginPath();

        ctx.moveTo(
            0,
            zeroY
        );

        ctx.lineTo(
            width,
            zeroY
        );

        ctx.stroke();

    }


    // Z1 = 0

    if (
        minZ1 <= 0 &&
        maxZ1 >= 0
    ) {

        const zeroX =
            z1ToX(0);

        ctx.beginPath();

        ctx.moveTo(
            zeroX,
            0
        );

        ctx.lineTo(
            zeroX,
            height
        );

        ctx.stroke();

    }


    // ========================================================
    // AXIS LABELS
    // ========================================================

    ctx.fillStyle = "#111";

    ctx.font = "bold 16px Arial";


    ctx.fillText(
        "Z1 →",
        width - 45,
        height - 10
    );


    ctx.fillText(
        "Z2 ↑",
        10,
        20
    );


    // ========================================================
    // SELECTED POINT
    // ========================================================

    ctx.fillStyle = "red";

    ctx.beginPath();

    ctx.arc(
        selectedX,
        selectedY,
        7,
        0,
        Math.PI * 2
    );

    ctx.fill();

}


// ============================================================
// CLICK LATENT SPACE
// ============================================================

canvas.addEventListener(
    "click",
    function(event) {

        const rect =
            canvas.getBoundingClientRect();


        const scaleX =
            canvas.width /
            rect.width;

        const scaleY =
            canvas.height /
            rect.height;


        const x =
            (event.clientX - rect.left)
            * scaleX;

        const y =
            (event.clientY - rect.top)
            * scaleY;


        // Convert click to Z1/Z2

        selectedZ1 =
            xToZ1(x);

        selectedZ2 =
            yToZ2(y);


        selectedX = x;
        selectedY = y;


        // Update displayed values

        document.getElementById(
            "z1Value"
        ).textContent =
            selectedZ1.toFixed(2);


        document.getElementById(
            "z2Value"
        ).textContent =
            selectedZ2.toFixed(2);


        document.getElementById(
            "latentVector"
        ).textContent =
            `[${selectedZ1.toFixed(2)}, ${selectedZ2.toFixed(2)}]`;


        drawGraph();


        console.log(
            "Selected:",
            selectedZ1,
            selectedZ2
        );

    }
);


// ============================================================
// GENERATE IMAGE
// ============================================================

async function generateImage() {

    const button =
        document.getElementById(
            "generateButton"
        );

    const imageContainer =
        document.getElementById(
            "imageContainer"
        );


    // Show loading

    button.disabled = true;

    button.textContent =
        "Generating...";


    imageContainer.innerHTML =
        `
        <div class="placeholder">
            Generating image...
        </div>
        `;


    try {

        // Send Z1 and Z2 to Flask

        const response =
            await fetch(
                "/generate",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        z1: selectedZ1,
                        z2: selectedZ2
                    })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.error ||
                "Image generation failed"
            );

        }


        // ====================================================
        // DISPLAY GENERATED IMAGE
        // ====================================================

        imageContainer.innerHTML =
            `
            <img
                src="data:image/png;base64,${data.image}"
                alt="Generated MNIST digit"
            >
            `;


        console.log(
            "Image generated successfully!"
        );


    } catch (error) {

        console.error(
            "Generation error:",
            error
        );


        imageContainer.innerHTML =
            `
            <div class="placeholder">
                Error generating image.
                <br><br>
                ${error.message}
            </div>
            `;

    }


    // Restore button

    button.disabled = false;

    button.textContent =
        "Generate Image";

}


// ============================================================
// GENERATE BUTTON
// ============================================================

document.getElementById(
    "generateButton"
).addEventListener(
    "click",
    generateImage
);


// ============================================================
// RANDOM LATENT POINT
// ============================================================

document.getElementById(
    "randomButton"
).addEventListener(
    "click",
    function() {

        // Generate random Z1/Z2
        // inside learned range

        const z1 =
            minZ1 +
            Math.random() *
            (maxZ1 - minZ1);

        const z2 =
            minZ2 +
            Math.random() *
            (maxZ2 - minZ2);


        selectedZ1 = z1;
        selectedZ2 = z2;


        selectedX =
            z1ToX(z1);

        selectedY =
            z2ToY(z2);


        // Update text

        document.getElementById(
            "z1Value"
        ).textContent =
            z1.toFixed(2);


        document.getElementById(
            "z2Value"
        ).textContent =
            z2.toFixed(2);


        document.getElementById(
            "latentVector"
        ).textContent =
            `[${z1.toFixed(2)}, ${z2.toFixed(2)}]`;


        drawGraph();


        // Automatically generate

        generateImage();

    }
);


// ============================================================
// INITIALIZE
// ============================================================

drawGraph();

loadLatentPoints();