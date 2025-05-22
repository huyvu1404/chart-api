const data = [];
for (const item of $input.all()) {
  for (const detail of item.json.details) {
    const date = detail.key_as_string;
    const day = (new Date(date).getDay() + 7) % 7;
    data.push({ date, day, sentiment: "Negative", doc_count: detail.doc_count });
  }
}
const filtered_data = data.slice(-14);

let last_week = []
let this_week = []
let last_week_started = false;
for (const entry of filtered_data) {
  if (entry.day === 0) last_week_started = true;
  if (last_week_started) {
    if (entry.day < 4) {
      last_week.push(entry.doc_count)
    } else {
      break;
    }
  }
}

for (let i = filtered_data.length - 1; i >= 0; i--) {
  const entry = filtered_data[i];
  if (entry.day === 0) {
    this_week.push(entry.doc_count);
    break;
  } else if (entry.day < 4) {
    this_week.push(entry.doc_count);
  }
}
this_week = this_week.slice().reverse()
return { "data": { title: "Negative Trend", 
                    x: ["Mo", "Tu", "We", "Th"], 
                    y: [last_week, this_week],
                    "labels": ["Last Week", "This Week"],
                    "colors": ["blue", "green"]
                  }};
