const fs = require("fs");
const path = require("path");

const folder = "./"; // current folder

// get all files
const files = fs.readdirSync(folder);
let data = []; 
files.forEach(file => {
    if (!file.endsWith(".json")) return;
    const filePath = path.join(folder, file);
    const raw = fs.readFileSync(filePath, "utf8");
    data.push(JSON.parse(raw));
});

fs.writeFileSync(path.join(folder,'message-1.json'), JSON.stringify(data, null, 2), "utf8");
