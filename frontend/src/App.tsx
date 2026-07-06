import { AlertTriangle, CalendarClock, LoaderCircle, Repeat2, Swords } from "lucide-react";
import { FormEvent, useEffect, useId, useMemo, useState } from "react";

const roles = ["top", "jungle", "mid", "bot", "support"] as const;
const sides = ["blue", "red"] as const;
const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

type Role = (typeof roles)[number];
type Side = (typeof sides)[number];
type RoleMap = Record<Role, string>;

type Metadata = {
  regions: string[];
  years: number[];
  splits: string[];
  stages: string[];
  contexts: MatchContext[];
  teams: string[];
  players: string[];
  champions: string[];
  model_version: string;
};

type MatchContext = {
  region: string;
  year: number;
  split: string;
  stage: string;
};

type DraftSide = {
  team_name: string;
  players: RoleMap;
  champions: RoleMap;
};

type DraftForm = {
  region: string;
  year: string;
  split: string;
  stage: string;
  first_pick_side: Side;
  blue: DraftSide;
  red: DraftSide;
};

type Prediction = {
  blue_win_probability: number;
  predicted_winner: Side;
  model_version: string;
  trained_at: string | null;
};

const emptyRoles = (): RoleMap =>
  roles.reduce((values, role) => ({ ...values, [role]: "" }), {} as RoleMap);

const initialForm: DraftForm = {
  region: "",
  year: String(new Date().getFullYear()),
  split: "",
  stage: "",
  first_pick_side: "blue",
  blue: { team_name: "", players: emptyRoles(), champions: emptyRoles() },
  red: { team_name: "", players: emptyRoles(), champions: emptyRoles() },
};

function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [form, setForm] = useState<DraftForm>(initialForm);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMetadata() {
      try {
        const response = await fetch(`${apiBaseUrl}/metadata`);
        if (!response.ok) {
          throw new Error(`Metadata request failed: ${response.status}`);
        }
        const nextMetadata = (await response.json()) as Metadata;
        setMetadata(nextMetadata);
        setForm(formWithDefaults(nextMetadata));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Metadata request failed");
      } finally {
        setLoading(false);
      }
    }

    loadMetadata();
  }, []);

  const blueProbability = prediction ? Math.round(prediction.blue_win_probability * 1000) / 10 : null;
  const redProbability = blueProbability === null ? null : Math.round((100 - blueProbability) * 10) / 10;
  const stageOptions = useMemo(() => stagesForContext(metadata, form.region, form.split), [
    metadata,
    form.region,
    form.split,
  ]);

  const canSubmit = useMemo(() => {
    return Boolean(
      form.region &&
        form.year &&
        form.split &&
        form.stage &&
        sides.every((side) => {
          const draftSide = form[side];
          return (
            draftSide.team_name &&
            roles.every((role) => draftSide.players[role] && draftSide.champions[role])
          );
        }),
    );
  }, [form]);

  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setPrediction(null);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, year: Number(form.year) }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail);
      }
      setPrediction(body as Prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction request failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="shell">
      <form className="console" onSubmit={submitPrediction}>
        <header className="topbar">
          <div>
            <h1>LoL Match Predictor</h1>
          </div>
          <div className="cutoff-chip">
            <CalendarClock size={18} aria-hidden="true" />
            <span>Information cutoff: Dec 2025</span>
          </div>
        </header>

        {error ? (
          <div className="notice" role="alert">
            <AlertTriangle size={18} aria-hidden="true" />
            <span>{error}</span>
          </div>
        ) : null}

        {loading ? (
          <div className="loading">
            <LoaderCircle className="spin" size={26} aria-hidden="true" />
            <span>Loading selectors</span>
          </div>
        ) : (
          <>
            <section className="match-strip" aria-label="Match context">
              <SelectField
                label="Region"
                value={form.region}
                options={metadata?.regions ?? []}
                onChange={(value) => setContextField("region", value, metadata, setForm)}
              />
              <SelectOrInput
                label="Year"
                value={form.year}
                options={(metadata?.years ?? []).map(String)}
                type="number"
                onChange={(value) => setField("year", value, setForm)}
              />
              <SelectOrInput
                label="Split"
                value={form.split}
                options={metadata?.splits ?? []}
                onChange={(value) => setContextField("split", value, metadata, setForm)}
              />
              <SelectOrInput
                label="Stage"
                value={form.stage}
                options={stageOptions}
                onChange={(value) => setField("stage", value, setForm)}
              />
              <SelectField
                label="First pick"
                value={form.first_pick_side}
                options={["blue", "red"]}
                onChange={(value) => setField("first_pick_side", value as Side, setForm)}
              />
            </section>

            <div className="draft-grid">
              <DraftSidePanel
                side="blue"
                value={form.blue}
                metadata={metadata}
                onChange={(nextSide) => setForm((current) => ({ ...current, blue: nextSide }))}
              />
              <button className="swap-button" type="button" onClick={() => swapSides(setForm)} title="Swap sides">
                <Repeat2 size={18} aria-hidden="true" />
                <span>Swap</span>
              </button>
              <DraftSidePanel
                side="red"
                value={form.red}
                metadata={metadata}
                onChange={(nextSide) => setForm((current) => ({ ...current, red: nextSide }))}
              />
            </div>

            <section className="prediction-band" aria-live="polite">
              <button className="submit-button" type="submit" disabled={!canSubmit || submitting}>
                {submitting ? <LoaderCircle className="spin" size={19} aria-hidden="true" /> : <Swords size={19} aria-hidden="true" />}
                <span>{submitting ? "Predicting" : "Predict"}</span>
              </button>

              {prediction && blueProbability !== null && redProbability !== null ? (
                <div className="result">
                  <div className="result-line">
                    <strong>{prediction.predicted_winner === "blue" ? "Blue" : "Red"} favored</strong>
                    <span>{blueProbability}% blue</span>
                  </div>
                  <div className="meter" aria-label={`Blue win probability ${blueProbability}%`}>
                    <span style={{ width: `${blueProbability}%` }} />
                  </div>
                  <div className="odds-row">
                    <span>Blue {blueProbability}%</span>
                    <span>Red {redProbability}%</span>
                  </div>
                </div>
              ) : (
                <div className="empty-result">Awaiting completed draft</div>
              )}
            </section>
          </>
        )}
      </form>
    </main>
  );
}

function formWithDefaults(metadata: Metadata): DraftForm {
  const firstPlayer = metadata.players[0] ?? "";
  const firstChampion = metadata.champions[0] ?? "";
  return {
    region: metadata.regions[0] ?? "",
    year: String(metadata.years[0] ?? new Date().getFullYear()),
    split: metadata.splits[0] ?? "",
    stage: metadata.stages[0] ?? "",
    first_pick_side: "blue",
    blue: {
      team_name: metadata.teams[0] ?? "",
      players: filledRoles(firstPlayer),
      champions: filledRoles(firstChampion),
    },
    red: {
      team_name: metadata.teams[1] ?? metadata.teams[0] ?? "",
      players: filledRoles(firstPlayer),
      champions: filledRoles(firstChampion),
    },
  };
}

function filledRoles(value: string): RoleMap {
  return roles.reduce((values, role) => ({ ...values, [role]: value }), {} as RoleMap);
}

function setField<Key extends keyof DraftForm>(
  key: Key,
  value: DraftForm[Key],
  setForm: React.Dispatch<React.SetStateAction<DraftForm>>,
) {
  setForm((current) => ({ ...current, [key]: value }));
}

function setContextField(
  key: "region" | "split",
  value: string,
  metadata: Metadata | null,
  setForm: React.Dispatch<React.SetStateAction<DraftForm>>,
) {
  setForm((current) => {
    const next = { ...current, [key]: value };
    const stages = stagesForContext(metadata, next.region, next.split);
    return stages.includes(next.stage) ? next : { ...next, stage: stages[0] ?? "" };
  });
}

function stagesForContext(metadata: Metadata | null, region: string, split: string): string[] {
  const stages =
    metadata?.contexts
      .filter((context) => context.region === region && context.split === split)
      .map((context) => context.stage) ?? [];
  return Array.from(new Set(stages)).sort();
}

function swapSides(setForm: React.Dispatch<React.SetStateAction<DraftForm>>) {
  setForm((current) => ({ ...current, blue: current.red, red: current.blue }));
}

type SelectFieldProps = {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
};

function SelectField({ label, value, options, onChange }: SelectFieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} required>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

type SelectOrInputProps = SelectFieldProps & {
  type?: "text" | "number";
};

function SelectOrInput({ label, value, options, type = "text", onChange }: SelectOrInputProps) {
  if (options.length > 0) {
    return <SelectField label={label} value={value} options={options} onChange={onChange} />;
  }

  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} required />
    </label>
  );
}

type DraftSidePanelProps = {
  side: Side;
  value: DraftSide;
  metadata: Metadata | null;
  onChange: (value: DraftSide) => void;
};

function DraftSidePanel({ side, value, metadata, onChange }: DraftSidePanelProps) {
  const title = side === "blue" ? "Blue Side" : "Red Side";

  function updateRole(kind: "players" | "champions", role: Role, nextValue: string) {
    onChange({
      ...value,
      [kind]: {
        ...value[kind],
        [role]: nextValue,
      },
    });
  }

  return (
    <section className={`side-panel ${side}`} aria-label={title}>
      <div className="side-header">
        <h2>{title}</h2>
        <ComboField
          label="Team"
          value={value.team_name}
          options={metadata?.teams ?? []}
          onChange={(team_name) => onChange({ ...value, team_name })}
        />
      </div>

      <div className="role-table">
        <div className="role-row heading" aria-hidden="true">
          <span>Role</span>
          <span>Player</span>
          <span>Champion</span>
        </div>
        {roles.map((role) => (
          <div className="role-row" key={role}>
            <span className="role-name">{role}</span>
            <ComboField
              label={`${title} ${role} player`}
              value={value.players[role]}
              options={metadata?.players ?? []}
              onChange={(nextValue) => updateRole("players", role, nextValue)}
            />
            <ComboField
              label={`${title} ${role} champion`}
              value={value.champions[role]}
              options={metadata?.champions ?? []}
              onChange={(nextValue) => updateRole("champions", role, nextValue)}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

function ComboField({ label, value, options, onChange }: SelectFieldProps) {
  const listId = useId();

  return (
    <label className="field">
      <span>{label}</span>
      <input
        list={listId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete="off"
        required
      />
      <datalist id={listId}>
        {options.map((option) => (
          <option key={option} value={option} />
        ))}
      </datalist>
    </label>
  );
}

export default App;
