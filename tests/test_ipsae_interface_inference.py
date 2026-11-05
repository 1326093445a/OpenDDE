# SPDX-License-Identifier: Apache-2.0
"""Selected interfaces survive featurization, both model paths and CIF relabeling."""

import copy
import json

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from opendde.data.inference import infer_dataloader
from opendde.metrics.ipsae import ipsae_summary, protein_token_groups
from opendde.model.opendde import OpenDDE
from runner.dumper import DataDumper
from tests.test_input_batching import _toy_config, _toy_features


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    class Features:
        entity_poly_type_and_seqs = {"entity_poly_type": {"1": "polypeptide(L)"}}

        def __init__(self, *args, **kwargs):
            pass

        def get_feature_dict(self):
            from opendde.data.core.featurizer import Featurizer

            feat = _toy_features(6, 101)
            feat["asym_id"] = torch.tensor([4, 4, 7, 7, 9, 9])
            atoms = AtomArray(24)
            atoms.chain_id = np.repeat(["A", "B", "C"], 8)
            atoms.atom_name = np.tile(["N", "CA", "C", "O"], 6)
            atoms.res_id = np.repeat(np.tile([1, 2], 3), 4)
            atoms.set_annotation("asym_id_int", np.repeat([4, 7, 9], 8))
            atoms.set_annotation("is_protein", np.ones(24, dtype=bool))
            atoms.set_annotation("distogram_rep_atom_mask", atoms.atom_name == "CA")
            feat["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
                atoms.atom_name
            )
            return feat, atoms, np.arange(6)

    monkeypatch.setattr(infer_dataloader, "SampleDictToFeatures", Features)
    monkeypatch.setattr(
        infer_dataloader.InferenceTemplateFeaturizer,
        "make_template_feature",
        lambda **kw: {},
    )
    cfg = _toy_config()
    cfg.dump_dir = str(tmp_path)
    cfg.need_atom_confidence = True
    jobs = [
        {
            "name": "grouped",
            "sequences": [{"proteinChain": {"sequence": "AA", "count": 3}}],
            "ipsae_interface": [["A"], ["B", "C"]],
        }
    ]
    return infer_dataloader.InferenceDataset(cfg, inputs=jobs)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("full_output", [False, True])
def test_selected_interface_reaches_model_and_writer(
    dataset, tmp_path, native, full_output
):
    data, atoms, error = dataset[0]
    assert not error, error
    feat = data["input_feature_dict"]
    assert feat["ipsae_interface"] == [["4"], ["7", "9"]]
    dataset.configs.need_atom_confidence = full_output
    model = OpenDDE(dataset.configs).eval()
    with torch.no_grad():
        if native:
            prediction = model.forward_batch([copy.deepcopy(feat)])[0]
        else:
            prediction = model(
                input_feature_dict=copy.deepcopy(feat), mode="inference"
            )[0]
    groups = protein_token_groups(atoms, feat["atom_to_token_idx"], 6)
    from opendde.model.sample_confidence import get_bin_params, logits_to_score

    pae = logits_to_score(
        prediction["pae"], **get_bin_params(dataset.configs.confidence.pae)
    )
    expected = [
        ipsae_summary(matrix, groups, interface=data["ipsae_interface"])
        for matrix in pae
    ]
    if not full_output:
        assert all("token_pair_pae" not in full for full in prediction["full_data"])
    dumper = DataDumper(str(tmp_path), need_atom_confidence=full_output)
    original = copy.deepcopy(prediction["summary_confidence"])
    dumper._save_confidence(
        prediction, str(tmp_path), "grouped", 101, None, atom_array=atoms
    )
    for i, reference in enumerate(expected):
        result = json.loads(
            (tmp_path / f"grouped_summary_confidence_sample_{i}.json").read_text()
        )
        assert result["ipsae_interface"] == reference["ipsae_interface"]
        assert result["ipsae_min_10"] == reference["ipsae_min_10"]
        assert (
            prediction["summary_confidence"][i]["ipsae_interface"]
            == original[i]["ipsae_interface"]
        )


def test_invalid_selected_chain_fails_during_preparation(dataset):
    dataset.inputs[0]["ipsae_interface"] = [["A"], ["missing"]]
    _, atoms, error = dataset[0]
    assert atoms is None
    assert "chain 'missing' has no protein CA tokens" in error
