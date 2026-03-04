function mockAnalyze(type) {
    const resultSection = document.getElementById("resultSection");
    resultSection.classList.remove("hidden");

    const confidence = Math.floor(Math.random() * 40) + 60;
    const offset = 377 - (377 * confidence) / 100;

    document.getElementById("progressCircle")
        .style.strokeDashoffset = offset;

    document.getElementById("confidenceText")
        .innerText = confidence + "%";

    document.getElementById("resultText")
        .innerText = type + " Result: " + (confidence > 75 ? "FAKE" : "REAL");

    window.scrollTo({ top: resultSection.offsetTop, behavior: 'smooth' });
}

/* Background particles */
const canvas = document.getElementById("particleCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particles = [];
for (let i = 0; i < 80; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 2,
        dx: Math.random() - 0.5,
        dy: Math.random() - 0.5
    });
}

function animateParticles() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = "#00f0ff";

    particles.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI*2);
        ctx.fill();

        p.x += p.dx;
        p.y += p.dy;

        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });

    requestAnimationFrame(animateParticles);
}

animateParticles();