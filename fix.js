const fs = require("fs");
const path = require("path");

const folder = "./"; 

const files = fs.readdirSync(folder);

files.forEach(file => {
  if (!file.endsWith(".json")) return;

  const filePath = path.join(folder, file);
  const raw = fs.readFileSync(filePath, "utf8");

  let data;
  try {
    data = JSON.parse(raw);
  } catch (err) {
    console.log(`Skipping invalid JSON: ${file}`);
    return;
  }

  const cleaned = data.messages
    .map(msg => {
      let content = msg.content ? Buffer.from(msg.content, "latin1").toString("utf8") : "";

      return {
        role: msg.sender_name === "Minh Tri\u00e1\u00ba\u00bft" ? "assistant" : "user",
        content: content.trim()
      };
    })
    .filter(msg =>
      msg.content &&
      msg.content !== "" &&
      !(/\.(jpg|jpeg|png|gif|webp)/i.test(msg.content))
    ).reverse();

  const newName = file.replace(".json", "-cleaned.json");
  const newPath = path.join(folder,'cleaned', newName);

  fs.writeFileSync(newPath, JSON.stringify(cleaned, null, 2), "utf8");

  console.log(`Processed: ${file} → ${newName}`);
});