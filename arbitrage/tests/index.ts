import { assert } from "chai";

describe.skip("legacy Anchor integration", () => {
  it("remains disabled until every protocol CPI is migrated", () => {
    assert.fail("legacy mainnet-facing test is intentionally disabled");
  });
});
