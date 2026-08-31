const token = "dummy_token"; // We just want to see if it resolves
const url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct";

async function test() {
    try {
        console.log("Fetching " + url);
        const res = await fetch(url, {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: "test" })
        });
        console.log("Status:", res.status);
        console.log("Body:", await res.text());
    } catch (e) {
        console.error("Fetch failed:", e);
    }
}
test();
